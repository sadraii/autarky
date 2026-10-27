//! TUI Dashboard module
//! Real-time benchmark progress display using ratatui

use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Alignment, Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Gauge, Paragraph, Row, Table},
    Frame, Terminal,
};
use std::io::{self, Stdout};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Once,
};
use std::time::{Duration, Instant};

use crate::hardware::HardwareInfo;
use crate::monitor::BenchmarkResult;

pub struct DashboardState {
    pub total: usize,
    pub completed: usize,
    pub current_ctx: usize,
    pub results: Vec<BenchmarkResult>,
    pub hardware: Option<HardwareInfo>,
    pub model_name: String,
    pub status_message: String,
    pub start_time: Option<Instant>,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            total: 0,
            completed: 0,
            current_ctx: 0,
            results: Vec::new(),
            hardware: None,
            model_name: String::new(),
            status_message: "Initializing...".to_string(),
            start_time: None,
        }
    }
}

pub struct Dashboard {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    pub state: DashboardState,
    guard: TerminalGuard,
}

impl Dashboard {
    pub fn new() -> anyhow::Result<Self> {
        let mut guard = TerminalGuard::acquire()?;
        // Restore even if enabling raw mode fails after a partial terminal change.
        guard.raw = true;
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        // Mark before writing: even a partial escape-sequence write needs cleanup.
        guard.alternate = true;
        execute!(stdout, EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;
        Ok(Self {
            terminal,
            state: DashboardState::default(),
            guard,
        })
    }

    pub fn set_total(&mut self, total: usize) {
        self.state.total = total;
    }

    pub fn set_model(&mut self, name: String) {
        self.state.model_name = name;
    }

    pub fn set_hardware(&mut self, hw: HardwareInfo) {
        self.state.hardware = Some(hw);
    }

    pub fn start_benchmark(&mut self, ctx_len: usize) {
        self.state.current_ctx = ctx_len;
        self.state.status_message = format!("Benchmarking context length {}...", ctx_len);
        if self.state.start_time.is_none() {
            self.state.start_time = Some(Instant::now());
        }
        let _ = self.draw();
    }

    pub fn complete_benchmark(&mut self, result: BenchmarkResult) {
        self.state.completed += 1;
        self.state.results.push(result);
        let _ = self.draw();
    }

    pub fn set_status(&mut self, msg: String) {
        self.state.status_message = msg;
        let _ = self.draw();
    }

    pub fn draw(&mut self) -> anyhow::Result<()> {
        // Extract state data before the closure to avoid borrow issues
        let total = self.state.total;
        let completed = self.state.completed;
        let current_ctx = self.state.current_ctx;
        let results = &self.state.results;
        let model_name = &self.state.model_name;
        let hw = self
            .state
            .hardware
            .as_ref()
            .map(|h| (h.gpu_family.as_str(), h.vram_gb));
        let elapsed = self.state.start_time.map_or(0, |t| t.elapsed().as_secs());
        let status_message = &self.state.status_message;

        self.terminal.draw(|f| {
            let size = f.area();

            // Main layout: header, progress, table, footer
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .margin(1)
                .constraints(
                    [
                        Constraint::Length(8), // Header
                        Constraint::Length(3), // Progress
                        Constraint::Min(5),    // Table
                        Constraint::Length(3), // Footer
                    ]
                    .as_ref(),
                )
                .split(size);

            // Header
            draw_header(f, chunks[0], model_name, hw, elapsed);

            // Progress
            draw_progress(f, chunks[1], completed, total, status_message);

            // Table
            draw_table(f, chunks[2], results, current_ctx, completed, total);

            // Footer
            draw_footer(f, chunks[3]);
        })?;
        Ok(())
    }

    pub fn cleanup(&mut self) -> anyhow::Result<()> {
        self.guard.restore()?;
        Ok(())
    }

    pub fn redraw(&mut self) -> anyhow::Result<()> {
        self.draw()
    }

    /// Nonblocking: the caller can select this every 50 ms during inference.
    pub fn check_quit(&self) -> bool {
        while event::poll(Duration::ZERO).unwrap_or(false) {
            match event::read() {
                Ok(event) if quit_event(&event) => return true,
                Ok(_) => {}
                Err(_) => return false,
            }
        }
        false
    }
}

static TERMINAL_ACTIVE: AtomicBool = AtomicBool::new(false);
static PANIC_HOOK: Once = Once::new();

struct TerminalGuard {
    raw: bool,
    alternate: bool,
}
impl TerminalGuard {
    fn acquire() -> io::Result<Self> {
        if TERMINAL_ACTIVE.swap(true, Ordering::SeqCst) {
            return Err(io::Error::other("A dashboard already owns the terminal"));
        }
        PANIC_HOOK.call_once(|| {
            let previous = std::panic::take_hook();
            std::panic::set_hook(Box::new(move |info| {
                if TERMINAL_ACTIVE.load(Ordering::SeqCst) {
                    let _ = restore_steps(
                        disable_raw_mode,
                        || execute!(io::stdout(), LeaveAlternateScreen),
                        || execute!(io::stdout(), crossterm::cursor::Show),
                    );
                }
                previous(info);
            }));
        });
        Ok(Self {
            raw: false,
            alternate: false,
        })
    }
    fn restore(&mut self) -> io::Result<()> {
        let result = restore_steps(
            || if self.raw { disable_raw_mode() } else { Ok(()) },
            || {
                if self.alternate {
                    execute!(io::stdout(), LeaveAlternateScreen)
                } else {
                    Ok(())
                }
            },
            || {
                if self.alternate {
                    execute!(io::stdout(), crossterm::cursor::Show)
                } else {
                    Ok(())
                }
            },
        );
        if result.is_ok() {
            self.raw = false;
            self.alternate = false;
        }
        result
    }
}
impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = self.restore();
        TERMINAL_ACTIVE.store(false, Ordering::SeqCst);
    }
}

fn restore_steps(
    raw: impl FnOnce() -> io::Result<()>,
    screen: impl FnOnce() -> io::Result<()>,
    cursor: impl FnOnce() -> io::Result<()>,
) -> io::Result<()> {
    // All steps must run, including when the first or second step fails.
    let raw = raw();
    let screen = screen();
    let cursor = cursor();
    raw.and(screen).and(cursor)
}

fn quit_event(event: &Event) -> bool {
    matches!(
        event,
        Event::Key(KeyEvent {
            kind: KeyEventKind::Press | KeyEventKind::Repeat,
            code: KeyCode::Char('q'),
            ..
        })
    ) || matches!(event, Event::Key(KeyEvent {
            kind: KeyEventKind::Press | KeyEventKind::Repeat, code: KeyCode::Char('c' | 'C'), modifiers, ..
        }) if modifiers.contains(KeyModifiers::CONTROL))
}

// Free functions for drawing widgets (avoiding borrow issues)

fn draw_header(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    model_name: &str,
    hw: Option<(&str, f64)>,
    elapsed: u64,
) {
    let hw_str = hw.map_or("Unknown".to_string(), |(family, vram)| {
        format!("{} ({:.1}GB VRAM)", family, vram)
    });

    let elapsed_str = format!("{}m{}s", elapsed / 60, elapsed % 60);

    let header_text = vec![
        Line::from(vec![
            Span::styled(
                "AUTARKY",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" — Local LLM Benchmark Dashboard"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Model: ", Style::default().fg(Color::Gray)),
            Span::styled(model_name, Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("GPU: ", Style::default().fg(Color::Gray)),
            Span::styled(hw_str, Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("Elapsed: ", Style::default().fg(Color::Gray)),
            Span::styled(elapsed_str, Style::default().fg(Color::Magenta)),
        ]),
    ];

    let header = Paragraph::new(header_text).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Benchmark Session"),
    );
    f.render_widget(header, area);
}

fn draw_progress(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    completed: usize,
    total: usize,
    status_message: &str,
) {
    let progress = if total > 0 {
        (completed as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title("Progress"))
        .gauge_style(Style::default().fg(Color::Cyan))
        .ratio(progress)
        .label(format!("{}/{} context lengths", completed, total));
    f.render_widget(gauge, area);

    // Status message below gauge
    let status = Paragraph::new(status_message)
        .style(Style::default().fg(Color::Yellow))
        .alignment(Alignment::Center);
    f.render_widget(status, area);
}

fn draw_table(
    f: &mut Frame,
    area: ratatui::layout::Rect,
    results: &[BenchmarkResult],
    current_ctx: usize,
    completed: usize,
    total: usize,
) {
    let header = Row::new(vec!["Ctx Len", "Status", "TPS", "Prompt/s", "Time"])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .height(1);

    let rows: Vec<Row> = results
        .iter()
        .map(|r| {
            let status = if r.status == "success" {
                Span::styled("✓ Success", Style::default().fg(Color::Green))
            } else {
                let label = match r.status.as_str() {
                    "oom" => "✗ OOM",
                    "cancelled" => "Cancelled",
                    _ => "Error",
                };
                Span::styled(label, Style::default().fg(Color::Red))
            };

            let tps = if r.status == "success" {
                format!("{:.1}", r.tokens_per_second)
            } else {
                "-".to_string()
            };

            let prompt = if r.status == "success" {
                format!("{:.1}", r.prompt_per_second)
            } else {
                "-".to_string()
            };

            Row::new(vec![
                Cell::from(r.context_length.to_string()),
                Cell::from(status),
                Cell::from(tps),
                Cell::from(prompt),
                Cell::from(""), // Time column placeholder
            ])
            .height(1)
        })
        .collect();

    // Current benchmarking row (if any)
    let mut all_rows = rows;
    if current_ctx > 0 && completed < total {
        let current_row = Row::new(vec![
            Cell::from(current_ctx.to_string()),
            Cell::from(Span::styled(
                "⏳ Running...",
                Style::default().fg(Color::Yellow),
            )),
            Cell::from(""),
            Cell::from(""),
            Cell::from(""),
        ])
        .style(Style::default().bg(Color::DarkGray))
        .height(1);
        all_rows.push(current_row);
    }

    let table = Table::new(
        all_rows,
        &[
            Constraint::Length(10),
            Constraint::Length(15),
            Constraint::Length(12),
            Constraint::Length(12),
            Constraint::Length(10),
        ],
    )
    .header(header)
    .block(Block::default().borders(Borders::ALL).title("Results"))
    .column_spacing(1);

    f.render_widget(table, area);
}

fn draw_footer(f: &mut Frame, area: ratatui::layout::Rect) {
    let footer_text = "Press q or Ctrl+C to quit | Autarky v0.1.0";
    let footer = Paragraph::new(footer_text)
        .style(Style::default().fg(Color::DarkGray))
        .alignment(Alignment::Center);
    f.render_widget(footer, area);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quit_keys_include_control_c_but_not_key_release() {
        for event in [
            KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE),
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
        ] {
            assert!(quit_event(&Event::Key(event)));
            assert!(!quit_event(&Event::Key(KeyEvent {
                kind: KeyEventKind::Release,
                ..event
            })));
        }
    }
    #[test]
    fn cleanup_attempts_screen_and_cursor_after_raw_mode_error() {
        let calls = std::cell::RefCell::new(Vec::new());
        let result = restore_steps(
            || {
                calls.borrow_mut().push("raw");
                Err(io::Error::other("fixture"))
            },
            || {
                calls.borrow_mut().push("screen");
                Ok(())
            },
            || {
                calls.borrow_mut().push("cursor");
                Ok(())
            },
        );
        assert!(result.is_err());
        assert_eq!(*calls.borrow(), ["raw", "screen", "cursor"]);
    }
    #[test]
    fn small_unicode_dashboard_renders_and_progress_is_bounded() {
        let backend = ratatui::backend::TestBackend::new(40, 18);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal
            .draw(|frame| {
                draw_header(
                    frame,
                    ratatui::layout::Rect::new(0, 0, 40, 8),
                    "模型🙂",
                    Some(("CPU", 0.0)),
                    0,
                );
                draw_progress(frame, ratatui::layout::Rect::new(0, 8, 40, 3), 3, 2, "done");
                draw_table(
                    frame,
                    ratatui::layout::Rect::new(0, 11, 40, 7),
                    &[],
                    0,
                    3,
                    2,
                );
            })
            .unwrap();
        let buffer = terminal.backend().buffer();
        assert_eq!(buffer.area.width, 40);
    }

    #[cfg(unix)]
    #[test]
    fn terminal_probe() {
        let Ok(mode) = std::env::var("AUTARKY_TUI_PROBE") else {
            return;
        };
        if mode == "partial" {
            let mut guard = TerminalGuard::acquire().unwrap();
            enable_raw_mode().unwrap();
            guard.raw = true;
            guard.alternate = true;
            execute!(io::stdout(), EnterAlternateScreen).unwrap();
            // Return before Terminal::new, as on a partial initialization error.
            return;
        }
        if mode == "panic" {
            let caught = std::panic::catch_unwind(|| {
                let _dashboard = Dashboard::new().unwrap();
                panic!("owned terminal panic fixture");
            });
            assert!(caught.is_err());
            return;
        }
        let mut dashboard = Dashboard::new().unwrap();
        dashboard.redraw().unwrap();
        let began = Instant::now();
        let mut quit = false;
        while began.elapsed() < Duration::from_secs(3) {
            if dashboard.check_quit() {
                quit = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        dashboard.cleanup().unwrap();
        dashboard.cleanup().unwrap();
        assert!(quit, "active dashboard must recognize q and Ctrl+C");
    }

    #[cfg(unix)]
    #[test]
    fn real_pty_restores_on_partial_init_panic_quit_and_control_c() {
        let script = r#"
import os, pty, select, subprocess, sys, termios, time
exe=sys.argv[1]
for mode in ['partial','panic','q','ctrl-c']:
    master,slave=pty.openpty()
    before=termios.tcgetattr(slave)
    env=dict(os.environ,AUTARKY_TUI_PROBE=mode)
    proc=subprocess.Popen([exe,'--exact','tui::tests::terminal_probe','--nocapture'],stdin=slave,stdout=slave,stderr=slave,env=env)
    output=b''; sent=False; began=time.monotonic()
    while proc.poll() is None and time.monotonic()-began<8:
        ready,_,_=select.select([master],[],[],0.02)
        if ready: output+=os.read(master,65536)
        if mode in ['q','ctrl-c'] and not sent and b'\x1b[?1049h' in output:
            os.write(master,b'q' if mode=='q' else b'\x03'); sent=True
    if proc.poll() is None:
        proc.kill();proc.wait();raise AssertionError('TUI probe hung: '+mode)
    while select.select([master],[],[],0)[0]: output+=os.read(master,65536)
    assert proc.returncode==0,(mode,output)
    assert termios.tcgetattr(slave)==before,('terminal flags not restored',mode)
    assert b'\x1b[?1049l' in output,('alternate screen not restored',mode,output)
    os.close(master);os.close(slave)
"#;
        let output = std::process::Command::new("python3")
            .arg("-c")
            .arg(script)
            .arg(std::env::current_exe().unwrap())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
