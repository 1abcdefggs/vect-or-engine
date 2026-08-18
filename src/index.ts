import { spawn, ChildProcess } from 'child_process';
import path from 'path';
import fs from 'fs';
import readline from 'readline';

export interface ValidationMarker {
  severity: 'Error' | 'Warning' | 'Info';
  message: string;
  line: number;
}

export interface ValidationResult {
  is_valid: boolean;
  extracted_value?: number;
  has_keyword_conflict: boolean;
  keyword_message?: string;
  value_message?: string;
  markers: ValidationMarker[];
}

export class EngineClient {
  private child: ChildProcess | null = null;
  private requestId = 0;
  private pendingRequests = new Map<string, { resolve: (res: any) => void; reject: (err: any) => void }>();
  private readlineInterface: readline.Interface | null = null;

  constructor() {
    this.initProcess();
  }

  private initProcess(): void {
    const exeName = process.platform === 'win32' ? 'vect-or-engine.exe' : 'vect-or-engine';

    const candidatePaths = [
      path.resolve(__dirname, '../target/release', exeName),
      path.resolve(__dirname, '../target/debug', exeName),
      path.resolve(__dirname, '../../vect-or-engine/target/release', exeName),
      path.resolve(__dirname, '../../vect-or-engine/target/debug', exeName),
      path.resolve(__dirname, '../../../vect-or-engine/target/release', exeName),
      path.resolve(__dirname, '../../../vect-or-engine/target/debug', exeName),
      path.resolve(process.cwd(), '../vect-or-engine/target/release', exeName),
      path.resolve(process.cwd(), '../vect-or-engine/target/debug', exeName),
    ];

    let binPath = '';
    for (const p of candidatePaths) {
      if (fs.existsSync(p)) {
        binPath = p;
        break;
      }
    }

    if (!binPath) {
      console.warn('[EngineClient] Compiled binary not found in candidate paths. Falling back to cargo run.');
      binPath = 'cargo';
    } else {
      console.log(`[EngineClient] Launching Rust engine binary: ${binPath}`);
    }

    const args = binPath === 'cargo' ? ['run', '--quiet', '--manifest-path', path.resolve(__dirname, '../Cargo.toml')] : [];

    this.child = spawn(binPath, args, {
      cwd: path.resolve(__dirname, '..'),
      stdio: ['pipe', 'pipe', 'pipe']
    });

    if (this.child.stdout) {
      this.readlineInterface = readline.createInterface({
        input: this.child.stdout,
        crlfDelay: Infinity
      });

      this.readlineInterface.on('line', (line: string) => {
        if (!line.trim()) return;
        try {
          const res = JSON.parse(line);
          const id = res.id;
          if (id && this.pendingRequests.has(id)) {
            const { resolve, reject } = this.pendingRequests.get(id)!;
            this.pendingRequests.delete(id);
            if (res.success) {
              resolve(res.data);
            } else {
              reject(new Error(res.error || 'Rust Engine Error'));
            }
          }
        } catch (e) {
          console.error('[EngineClient] Failed to parse daemon line:', line, e);
        }
      });
    }

    this.child.on('exit', (code) => {
      console.warn(`[EngineClient] Rust daemon exited with code ${code}`);
      this.child = null;
    });

    this.child.on('error', (err) => {
      console.error('[EngineClient] Rust daemon error:', err);
    });
  }
  private async sendRequest(method: string, params: any = {}): Promise<any> {
    if (!this.child || !this.child.stdin) {
      this.initProcess();
    }

    const id = String(++this.requestId);
    const payload = JSON.stringify({ method, params, id }) + '\n';

    return new Promise((resolve, reject) => {
      this.pendingRequests.set(id, { resolve, reject });
      this.child!.stdin!.write(payload, 'utf-8', (err) => {
        if (err) {
          this.pendingRequests.delete(id);
          reject(err);
        }
      });
    });
  }

  public async loadProfile(profilePath: string): Promise<any> {
    return this.sendRequest('load_profile', { path: profilePath });
  }

  public async loadKnowledgeBase(kbPath: string): Promise<any> {
    return this.sendRequest('load_knowledge_base', { path: kbPath });
  }

  public async validateDocument(text: string): Promise<ValidationResult> {
    return this.sendRequest('validate', { text });
  }

  public async translateColloquial(text: string): Promise<any> {
    return this.sendRequest('translate', { text });
  }

  public async searchVector(vector: number[], topK: number = 5): Promise<any[]> {
    return this.sendRequest('search', { vector, top_k: topK });
  }
}
