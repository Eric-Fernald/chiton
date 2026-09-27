package security

import (
	"context"
	"testing"
)

// False-positive regression tests: each FP input must yield zero findings while a genuine true
// positive still fires.

func TestInsecureTLSVerify_FP_PythonVerifyFalse(t *testing.T) {
	r := InsecureTLSVerify{}

	// False positives: verify=False outside any HTTP-client context.
	fp := []string{
		`df = pandas.read_csv(path, verify=False)`,
		`verify = fields.Bool(verify=False)`,
		`schema.verify = False`,
		`# r = requests.get(url, verify=False)  # disabled for now`,
	}
	for _, src := range fp {
		if got := r.MatchFile(context.Background(), target(t, "x.py", src)); len(got) != 0 {
			t.Fatalf("expected no finding for FP %q, got %d", src, len(got))
		}
	}

	// True positives: verify=False inside a real HTTP-client context still fires.
	tp := []string{
		`r = requests.get(url, verify=False)`,
		`r = httpx.post(url, json=data, verify=False)`,
		`resp = session.request("GET", url, verify=False)`,
		`r = client.get(url, verify=False)`,
	}
	for _, src := range tp {
		if got := r.MatchFile(context.Background(), target(t, "x.py", src)); len(got) != 1 {
			t.Fatalf("expected 1 finding for TP %q, got %d", src, len(got))
		}
	}
}

func TestCommandInjection_FP_ExecSync(t *testing.T) {
	r := CommandInjection{}

	// False positives: execSync on an unrelated object (e.g. better-sqlite3).
	fp := []string{
		`db.prepare(sql).execSync()`,
		`stmt.execSync()`,
		`const rows = query.execSync(params)`,
	}
	for _, src := range fp {
		if got := r.MatchFile(context.Background(), target(t, "x.js", src)); len(got) != 0 {
			t.Fatalf("expected no finding for FP %q, got %d", src, len(got))
		}
	}

	// True positives: execSync on the child_process module still fires.
	tp := []string{
		`child_process.execSync("ls " + dir)`,
		`cp.execSync(cmd)`,
	}
	for _, src := range tp {
		if got := r.MatchFile(context.Background(), target(t, "x.js", src)); len(got) != 1 {
			t.Fatalf("expected 1 finding for TP %q, got %d", src, len(got))
		}
	}
}

func TestCORSWildcard_FP_OriginSubstring(t *testing.T) {
	r := CORSWildcard{}

	// False positives: an identifier ending in origin/origins (not the key).
	fp := []string{
		`myorigin = "*"`,
		`isolatedOrigins = "*"`,
	}
	for _, src := range fp {
		if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 0 {
			t.Fatalf("expected no finding for FP %q, got %d", src, len(got))
		}
	}

	// True positives: the actual origin(s) config keys still fire.
	tp := []string{
		`origin = "*"`,
		`origins = "*"`,
		`allowed_origins = "*"`,
		`CORS_ALLOWED_ORIGINS = "*"`,
	}
	for _, src := range tp {
		if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 1 {
			t.Fatalf("expected 1 finding for TP %q, got %d", src, len(got))
		}
	}
}
