package security

import (
	"context"
	"testing"
)

// These tests reuse the package-local target() helper defined in security_test.go.

func TestInsecureTLSVerify(t *testing.T) {
	pos := map[string]string{
		"go":  `cfg := &tls.Config{InsecureSkipVerify: true}`,
		"py":  `r = requests.get(url, verify=False)`,
		"ssl": `ctx = ssl._create_unverified_context()`,
		"js":  `const a = new https.Agent({ rejectUnauthorized: false })`,
	}
	r := InsecureTLSVerify{}
	for name, src := range pos {
		t.Run("pos/"+name, func(t *testing.T) {
			if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 1 {
				t.Fatalf("expected 1 finding for %q, got %d", src, len(got))
			}
		})
	}
	neg := []string{
		`cfg := &tls.Config{InsecureSkipVerify: false}`,
		`// InsecureSkipVerify: true would be bad here`,
		`r = requests.get(url, verify=True)`,
		`const a = new https.Agent({ rejectUnauthorized: true })`,
	}
	for _, src := range neg {
		if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 0 {
			t.Fatalf("expected no finding for %q, got %d", src, len(got))
		}
	}
}

func TestWeakHash(t *testing.T) {
	pos := map[string]string{
		"go-md5":  `h := md5.New()`,
		"go-sha1": `s := sha1.Sum(b)`,
		"py":      `d = hashlib.md5(data).hexdigest()`,
		"js":      `const h = crypto.createHash("sha1")`,
		"java":    `MessageDigest md = MessageDigest.getInstance("MD5");`,
		"java-s1": `MessageDigest md = MessageDigest.getInstance("SHA-1");`,
	}
	r := WeakHash{}
	for name, src := range pos {
		t.Run("pos/"+name, func(t *testing.T) {
			if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 1 {
				t.Fatalf("expected 1 finding for %q, got %d", src, len(got))
			}
		})
	}
	neg := []string{
		`h := sha256.New()`,
		`const h = crypto.createHash("sha256")`,
		`MessageDigest md = MessageDigest.getInstance("SHA-256");`,
		`// md5.New() is no longer used`,
	}
	for _, src := range neg {
		if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 0 {
			t.Fatalf("expected no finding for %q, got %d", src, len(got))
		}
	}
}

func TestCommandInjection(t *testing.T) {
	pos := map[string]string{
		"py-shell":  `subprocess.run(cmd, shell=True)`,
		"py-system": `os.system("rm -rf " + path)`,
		"js-exec":   `child_process.exec("ls " + dir)`,
		"go-sh":     `exec.Command("sh", "-c", userCmd)`,
		"go-bash":   `exec.Command("bash", "-c", userCmd)`,
	}
	r := CommandInjection{}
	for name, src := range pos {
		t.Run("pos/"+name, func(t *testing.T) {
			if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 1 {
				t.Fatalf("expected 1 finding for %q, got %d", src, len(got))
			}
		})
	}
	neg := []string{
		`subprocess.run(cmd, shell=False)`,
		`out, err := exec.Command("ls", "-l").Output()`,
		`child_process.execFile("ls", [dir])`,
		`# os.system(x) was removed`,
	}
	for _, src := range neg {
		if got := r.MatchFile(context.Background(), target(t, "x.py", src)); len(got) != 0 {
			t.Fatalf("expected no finding for %q, got %d", src, len(got))
		}
	}
}

func TestCORSWildcard(t *testing.T) {
	pos := []string{
		`w.Header().Set("Access-Control-Allow-Origin", "*")`,
		`{"Access-Control-Allow-Origin": "*"}`,
		`CORS_ALLOWED_ORIGINS = "*"`,
		`app.use(cors({ origin: "*" }))`,
	}
	r := CORSWildcard{}
	for _, src := range pos {
		if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 1 {
			t.Fatalf("expected 1 finding for %q, got %d", src, len(got))
		}
	}
	neg := []string{
		`w.Header().Set("Access-Control-Allow-Origin", "https://app.example.com")`,
		`CORS_ALLOWED_ORIGINS = "https://example.com"`,
		`// Access-Control-Allow-Origin: * is dangerous`,
	}
	for _, src := range neg {
		if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 0 {
			t.Fatalf("expected no finding for %q, got %d", src, len(got))
		}
	}
}

func TestYAMLUnsafeLoad(t *testing.T) {
	r := YAMLUnsafeLoad{}
	if got := r.MatchFile(context.Background(), target(t, "x.py", `data = yaml.load(f)`)); len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d", len(got))
	}
	neg := []string{
		`data = yaml.safe_load(f)`,
		`data = yaml.load(f, Loader=yaml.SafeLoader)`,
		`# yaml.load(f) was unsafe`,
	}
	for _, src := range neg {
		if got := r.MatchFile(context.Background(), target(t, "x.py", src)); len(got) != 0 {
			t.Fatalf("expected no finding for %q, got %d", src, len(got))
		}
	}
}

func TestDebugModeEnabled(t *testing.T) {
	pos := []string{
		`app.run(debug=True)`,
		`DEBUG = True`,
		`{ "debug": true }`,
	}
	r := DebugModeEnabled{}
	for _, src := range pos {
		if got := r.MatchFile(context.Background(), target(t, "app.py", src)); len(got) != 1 {
			t.Fatalf("expected 1 finding for %q, got %d", src, len(got))
		}
	}
	neg := []string{
		`app.run(debug=False)`,
		`DEBUG = False`,
		`# DEBUG = True in dev only`,
	}
	for _, src := range neg {
		if got := r.MatchFile(context.Background(), target(t, "app.py", src)); len(got) != 0 {
			t.Fatalf("expected no finding for %q, got %d", src, len(got))
		}
	}
	// Test files are skipped.
	if got := r.MatchFile(context.Background(), target(t, "test_app.py", `DEBUG = True`)); len(got) != 0 {
		t.Fatalf("expected test file to be skipped, got %d", len(got))
	}
}

func TestBindAllInterfaces(t *testing.T) {
	pos := []string{
		`srv.Listen("0.0.0.0:8080")`,
		`host = "0.0.0.0"`,
		`app.run(host='0.0.0.0', port=5000)`,
	}
	r := BindAllInterfaces{}
	for _, src := range pos {
		if got := r.MatchFile(context.Background(), target(t, "main.go", src)); len(got) != 1 {
			t.Fatalf("expected 1 finding for %q, got %d", src, len(got))
		}
	}
	neg := []string{
		`host = "127.0.0.1"`,
		`version = "1.0.0.0"`,
		`// bind to 0.0.0.0 in containers`,
	}
	for _, src := range neg {
		if got := r.MatchFile(context.Background(), target(t, "main.go", src)); len(got) != 0 {
			t.Fatalf("expected no finding for %q, got %d", src, len(got))
		}
	}
}

func TestJWTNoneAlg(t *testing.T) {
	pos := []string{
		`{"alg":"none"}`,
		`token = jwt.encode(payload, None, algorithm="none")`,
		`{ algorithm: "none" }`,
		`verifyOpts := jwt.Options{algorithms: ["none"]}`,
	}
	r := JWTNoneAlg{}
	for _, src := range pos {
		if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 1 {
			t.Fatalf("expected 1 finding for %q, got %d", src, len(got))
		}
	}
	neg := []string{
		`{"alg":"HS256"}`,
		`algorithm = "RS256"`,
		`// alg: "none" must never be allowed`,
	}
	for _, src := range neg {
		if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 0 {
			t.Fatalf("expected no finding for %q, got %d", src, len(got))
		}
	}
}

func TestCredentialsInURL(t *testing.T) {
	pos := []string{
		`db := "postgres://admin:s3cret@db.example.com:5432/app"`,
		`url = "https://user:pass@example.com/path"`,
		`redis://default:hunter2@cache:6379`,
	}
	r := CredentialsInURL{}
	for _, src := range pos {
		if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 1 {
			t.Fatalf("expected 1 finding for %q, got %d", src, len(got))
		}
	}
	neg := []string{
		`url = "https://example.com/path"`,
		`host := "db.example.com:5432"`,
		`git@github.com:org/repo.git`,
	}
	for _, src := range neg {
		if got := r.MatchFile(context.Background(), target(t, "x.go", src)); len(got) != 0 {
			t.Fatalf("expected no finding for %q, got %d", src, len(got))
		}
	}
}
