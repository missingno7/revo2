use crate::{
    evolution::{self, Config, Engine, Stats},
    export::{self, HistoryPoint},
    program::Program,
    target::Target,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    io::Read,
    sync::{mpsc, Arc, Mutex},
    thread,
    time::{Duration, Instant},
};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

#[derive(Clone, Serialize)]
struct View {
    config: Config,
    backend: String,
    backend_key: String,
    target_name: String,
    stats: Stats,
    running: bool,
    selected: usize,
    following_best: bool,
    program: Program,
    active_nodes: Vec<usize>,
    scores: Vec<f32>,
    history: Vec<HistoryPoint>,
    elapsed_seconds: f64,
    generations_per_second: f64,
    validation_mse: f32,
    revision: u64,
    error: Option<String>,
}

#[derive(Clone)]
struct Shared {
    view: View,
    target: Vec<u8>,
    best: Vec<u8>,
    selected: Vec<u8>,
    best_program: Program,
}

#[derive(Deserialize)]
struct Action {
    action: String,
    config: Option<Config>,
    backend: Option<String>,
    preset: Option<String>,
    index: Option<usize>,
}
enum Command {
    Action(Action),
    Upload(Target),
}

struct Session {
    engine: Box<dyn Engine>,
    config: Config,
    backend: String,
    target: Target,
    running: bool,
    selected: Option<usize>,
    history: Vec<HistoryPoint>,
    evolution_seconds: f64,
    revision: u64,
}

impl Session {
    fn new(config: Config, backend: String, target: Target) -> Result<Self> {
        config.validate()?;
        let engine =
            evolution::create(&backend, &config, target.samples(config.sample_resolution))?;
        Ok(Self {
            engine,
            config,
            backend,
            target,
            running: false,
            selected: None,
            history: Vec::new(),
            evolution_seconds: 0.0,
            revision: 0,
        })
    }

    fn advance(&mut self, count: u32) -> Result<()> {
        let start = Instant::now();
        self.engine.step(count)?;
        self.evolution_seconds += start.elapsed().as_secs_f64();
        Ok(())
    }

    fn snapshot(&mut self, record: bool) -> Result<Shared> {
        let stats = self.engine.stats()?;
        if record
            && self
                .history
                .last()
                .is_none_or(|p| p.generation != stats.generation)
        {
            self.history.push(HistoryPoint {
                generation: stats.generation,
                best: stats.best_score,
                mean: stats.mean_score,
                seconds: self.evolution_seconds,
            });
            // Decimate history to preserve the whole run without unbounded memory.
            if self.history.len() > 2048 {
                self.history = self.history.iter().step_by(2).cloned().collect();
            }
        }
        let selected = self.selected.unwrap_or(stats.best_index);
        let program = self.engine.program(selected)?;
        let best_program = if selected == stats.best_index {
            program.clone()
        } else {
            self.engine.program(stats.best_index)?
        };
        let active = program.active_mask();
        let validation_mse = crate::target::mse(&best_program, &self.target.samples(64));
        self.revision += 1;
        Ok(Shared {
            target: export::png(&self.target.image)?,
            best: export::png(&self.engine.render(stats.best_index, 256, 256)?)?,
            selected: export::png(&self.engine.render(selected, 256, 256)?)?,
            best_program,
            view: View {
                config: self.config.clone(),
                backend: self.engine.backend().into(),
                backend_key: self.backend.clone(),
                target_name: self.target.name.clone(),
                running: self.running,
                selected,
                following_best: self.selected.is_none(),
                active_nodes: (0..program.nodes.len())
                    .filter(|i| active & (1u64 << i) != 0)
                    .collect(),
                scores: self.engine.fitness()?,
                program,
                history: self.history.clone(),
                elapsed_seconds: self.evolution_seconds,
                generations_per_second: if self.evolution_seconds > 0.0 {
                    stats.generation as f64 / self.evolution_seconds
                } else {
                    0.0
                },
                stats,
                validation_mse,
                revision: self.revision,
                error: None,
            },
        })
    }

    fn command(&mut self, command: Command) -> Result<()> {
        match command {
            Command::Upload(target) => {
                let mut next = Self::new(self.config.clone(), self.backend.clone(), target)?;
                next.revision = self.revision;
                *self = next;
            }
            Command::Action(a) => match a.action.as_str() {
                "start" => self.running = true,
                "pause" => self.running = false,
                "step" => {
                    self.running = false;
                    self.advance(1)?;
                }
                "select" => {
                    anyhow::ensure!(
                        a.index.is_some_and(|i| i < self.config.population()),
                        "invalid cell index"
                    );
                    self.selected = a.index;
                }
                "follow" => self.selected = None,
                "reset" => {
                    let config = a.config.unwrap_or_else(|| self.config.clone());
                    let backend = a.backend.unwrap_or_else(|| self.backend.clone());
                    let target = match a.preset {
                        Some(name) => Target::preset(&name, 512)?,
                        None => self.target.clone(),
                    };
                    // Build the replacement before touching the current session.
                    let mut next = Self::new(config, backend, target)?;
                    next.revision = self.revision;
                    *self = next;
                }
                _ => anyhow::bail!("unknown action"),
            },
        }
        Ok(())
    }
}

fn worker(mut session: Session, receiver: mpsc::Receiver<Command>, shared: Arc<Mutex<Shared>>) {
    let mut last = Instant::now();
    loop {
        let command = if session.running {
            receiver.try_recv().map(Some).or_else(|e| match e {
                mpsc::TryRecvError::Empty => Ok(None),
                mpsc::TryRecvError::Disconnected => Err(()),
            })
        } else {
            receiver.recv().map(Some).map_err(|_| ())
        };
        let command = match command {
            Ok(c) => c,
            Err(()) => break,
        };
        let mut changed = command.is_some();
        let result = if let Some(command) = command {
            session.command(command)
        } else {
            let count = if session.backend == "cuda" { 8 } else { 1 };
            session.advance(count)
        };
        if let Err(error) = result {
            session.running = false;
            let mut state = shared.lock().unwrap();
            state.view.running = false;
            state.view.error = Some(format!("{error:#}"));
            continue;
        }
        changed |= last.elapsed() > Duration::from_millis(250);
        if changed {
            match session.snapshot(true) {
                Ok(snapshot) => *shared.lock().unwrap() = snapshot,
                Err(error) => {
                    session.running = false;
                    let mut state = shared.lock().unwrap();
                    state.view.running = false;
                    state.view.error = Some(format!("{error:#}"));
                }
            }
            last = Instant::now();
        }
    }
}

fn respond(request: Request, status: u16, mime: &str, body: Vec<u8>) {
    let response = Response::from_data(body)
        .with_status_code(StatusCode(status))
        .with_header(Header::from_bytes("Content-Type", mime).unwrap())
        .with_header(Header::from_bytes("Cache-Control", "no-store").unwrap())
        .with_header(Header::from_bytes("X-Content-Type-Options", "nosniff").unwrap());
    let _ = request.respond(response);
}

fn handle(
    mut request: Request,
    shared: &Arc<Mutex<Shared>>,
    sender: &mpsc::SyncSender<Command>,
    port: u16,
) -> Result<()> {
    let authority = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Host"))
        .map(|h| h.value.as_str())
        .unwrap_or("");
    if authority != format!("127.0.0.1:{port}") && authority != format!("localhost:{port}") {
        respond(
            request,
            403,
            "text/plain",
            b"Only localhost requests are accepted".to_vec(),
        );
        return Ok(());
    }
    let path = request.url().split('?').next().unwrap_or("/").to_string();
    if request.method() == &Method::Get {
        match path.as_str() {
            "/" => respond(
                request,
                200,
                "text/html; charset=utf-8",
                include_bytes!("../web/index.html").to_vec(),
            ),
            "/app.css" => respond(
                request,
                200,
                "text/css",
                include_bytes!("../web/app.css").to_vec(),
            ),
            "/app.js" => respond(
                request,
                200,
                "text/javascript",
                include_bytes!("../web/app.js").to_vec(),
            ),
            "/api/state" => respond(
                request,
                200,
                "application/json",
                serde_json::to_vec(&shared.lock().unwrap().view)?,
            ),
            "/target.png" | "/best.png" | "/selected.png" => {
                let state = shared.lock().unwrap();
                let bytes = match path.as_str() {
                    "/target.png" => &state.target,
                    "/best.png" => &state.best,
                    _ => &state.selected,
                }
                .clone();
                drop(state);
                respond(request, 200, "image/png", bytes);
            }
            "/program.json" => respond(
                request,
                200,
                "application/json",
                serde_json::to_vec_pretty(&shared.lock().unwrap().best_program)?,
            ),
            "/program.glsl" => respond(
                request,
                200,
                "text/plain; charset=utf-8",
                shared.lock().unwrap().best_program.glsl().into_bytes(),
            ),
            _ => respond(request, 404, "text/plain", b"Not found".to_vec()),
        }
        return Ok(());
    }
    if request.method() != &Method::Post {
        respond(request, 405, "text/plain", b"Method not allowed".to_vec());
        return Ok(());
    }
    // Only same-origin browser controls may mutate the local experiment.
    let host = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Host"))
        .map(|h| h.value.as_str())
        .unwrap_or("");
    let origin = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Origin"))
        .map(|h| h.value.as_str());
    if origin.is_some_and(|o| o != format!("http://{host}")) {
        respond(
            request,
            403,
            "text/plain",
            b"Cross-origin request rejected".to_vec(),
        );
        return Ok(());
    }
    let content_type = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Content-Type"))
        .map(|h| h.value.as_str())
        .unwrap_or("");
    let valid_type = match path.as_str() {
        "/api/action" => content_type.starts_with("application/json"),
        "/api/target" => content_type == "application/octet-stream",
        _ => false,
    };
    if !valid_type {
        respond(request, 415, "text/plain", b"Unsupported request".to_vec());
        return Ok(());
    }
    let limit = if path == "/api/target" {
        10 * 1024 * 1024
    } else {
        16 * 1024
    };
    let mut bytes = Vec::new();
    request
        .as_reader()
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        respond(request, 413, "text/plain", b"Request too large".to_vec());
        return Ok(());
    }
    let command: Result<_> = if path == "/api/action" {
        serde_json::from_slice(&bytes)
            .map(Command::Action)
            .map_err(Into::into)
    } else {
        Target::from_bytes("Uploaded image", &bytes).map(Command::Upload)
    };
    match command {
        Ok(c) => match sender.try_send(c) {
            Ok(()) => respond(
                request,
                202,
                "application/json",
                b"{\"accepted\":true}".to_vec(),
            ),
            Err(_) => respond(
                request,
                503,
                "text/plain",
                b"Experiment is busy; try again shortly".to_vec(),
            ),
        },
        Err(error) => respond(request, 400, "text/plain", error.to_string().into_bytes()),
    }
    Ok(())
}

pub fn serve(port: u16, backend: &str) -> Result<()> {
    let server = Server::http(("127.0.0.1", port))
        .map_err(|e| anyhow::anyhow!("binding local playground: {e}"))?;
    let mut session = Session::new(
        Config::default(),
        backend.into(),
        Target::preset("orbit", 512)?,
    )?;
    let initial = session.snapshot(true)?;
    println!(
        "{}\nPlayground: http://127.0.0.1:{port}\nPress Ctrl+C to stop.",
        session.engine.backend()
    );
    let shared = Arc::new(Mutex::new(initial));
    let (sender, receiver) = mpsc::sync_channel(16);
    let worker_state = shared.clone();
    thread::spawn(move || worker(session, receiver, worker_state));
    for request in server.incoming_requests() {
        if let Err(error) = handle(request, &shared, &sender, port) {
            eprintln!("HTTP request: {error:#}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        Session::new(
            Config {
                width: 2,
                height: 2,
                nodes: 3,
                sample_resolution: 4,
                ..Default::default()
            },
            "cpu".into(),
            Target::preset("orbit", 16).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn failed_reset_preserves_the_existing_experiment() {
        let mut s = session();
        s.advance(3).unwrap();
        let before = s.engine.fitness().unwrap();
        let invalid = Config {
            sample_resolution: u32::MAX,
            ..s.config.clone()
        };
        let action = Action {
            action: "reset".into(),
            config: Some(invalid),
            backend: None,
            preset: Some("rings".into()),
            index: None,
        };
        assert!(s.command(Command::Action(action)).is_err());
        assert_eq!(s.engine.stats().unwrap().generation, 3);
        assert_eq!(s.engine.fitness().unwrap(), before);
        assert_eq!(s.target.name, "orbit");
    }

    #[test]
    fn target_upload_restarts_population_and_history() {
        let mut s = session();
        s.advance(2).unwrap();
        s.command(Command::Upload(Target::preset("rings", 16).unwrap()))
            .unwrap();
        assert_eq!(s.engine.stats().unwrap().generation, 0);
        assert_eq!(s.target.name, "rings");
        assert!(s.history.is_empty());
        assert!(!s.running);
    }
}
