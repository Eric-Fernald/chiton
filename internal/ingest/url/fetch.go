package url

import (
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// maxBodyBytes caps how much of a response/file is read into memory.
const maxBodyBytes = 32 << 20 // 32 MiB

// fetchResult is the raw payload a fetcher returns before text extraction.
type fetchResult struct {
	canonicalURL string // final URL after redirects, or file:// for local paths
	contentType  string // MIME type (lowercased, parameters stripped)
	title        string // best title known pre-extraction (filename / yt-dlp)
	ext          string // lowercased file extension hint without the dot
	body         []byte
}

// fetcher retrieves a URL's bytes. Isolated so tests can inject canned payloads
// and so YouTube/social handling stays out of the node-building path.
type fetcher interface {
	fetch(ctx context.Context, rawURL string) (*fetchResult, error)
}

// httpFetcher fetches over HTTP(S), reads local files (bare paths or file://),
// and delegates YouTube URLs to yt-dlp.
type httpFetcher struct {
	client *http.Client
	// ytdlp overrides the yt-dlp binary name (default "yt-dlp"); for tests.
	ytdlp string
}

func (h *httpFetcher) fetch(ctx context.Context, rawURL string) (*fetchResult, error) {
	u, err := url.Parse(rawURL)
	if err != nil || u.Scheme == "" {
		// Treat an unparseable or scheme-less string as a local path.
		return h.fetchFile(rawURL)
	}
	switch strings.ToLower(u.Scheme) {
	case "file":
		path, ok := model.PathFromFileURI(rawURL)
		if !ok { // without a lower-case "file://" prefix (e.g. "FILE://x"), the raw string is the path
			path = filepath.FromSlash(rawURL)
		}
		return h.fetchFile(path)
	case "http", "https":
		if isYouTube(u.Host) {
			return h.fetchYouTube(ctx, rawURL)
		}
		return h.fetchHTTP(ctx, rawURL)
	default:
		return nil, fmt.Errorf("unsupported URL scheme %q", u.Scheme)
	}
}

func (h *httpFetcher) fetchFile(path string) (*fetchResult, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		abs = path
	}
	f, err := os.Open(abs)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	body, err := io.ReadAll(io.LimitReader(f, maxBodyBytes))
	if err != nil {
		return nil, err
	}
	ext := strings.TrimPrefix(strings.ToLower(filepath.Ext(abs)), ".")
	return &fetchResult{
		canonicalURL: model.FileURI(abs), // abs is already absolute, so FileURI only cleans it
		contentType:  contentTypeForExt(ext),
		title:        strings.TrimSuffix(filepath.Base(abs), filepath.Ext(abs)),
		ext:          ext,
		body:         body,
	}, nil
}

func (h *httpFetcher) fetchHTTP(ctx context.Context, rawURL string) (*fetchResult, error) {
	if u, err := url.Parse(rawURL); err == nil {
		if err := hostAllowed(u.Hostname()); err != nil {
			return nil, err
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	// A desktop UA gets readable HTML from most sites that gate bare clients.
	req.Header.Set("User-Agent", "Mozilla/5.0 (compatible; chiton/0.1; +https://github.com/Eric-Fernald/chiton)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.8")

	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("HTTP %d for %s", resp.StatusCode, rawURL)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	if err != nil {
		return nil, err
	}
	ct := strings.ToLower(strings.TrimSpace(resp.Header.Get("Content-Type")))
	if i := strings.IndexByte(ct, ';'); i >= 0 {
		ct = strings.TrimSpace(ct[:i])
	}
	final := rawURL
	if resp.Request != nil && resp.Request.URL != nil {
		final = resp.Request.URL.String()
	}
	ext := ""
	if u, e := url.Parse(final); e == nil {
		ext = strings.TrimPrefix(strings.ToLower(filepath.Ext(u.Path)), ".")
	}
	return &fetchResult{
		canonicalURL: final,
		contentType:  ct,
		ext:          ext,
		body:         body,
	}, nil
}

// fetchYouTube shells out to yt-dlp for a video's title and English (auto-)subtitles as WebVTT,
// returning a clear error when yt-dlp is not installed.
func (h *httpFetcher) fetchYouTube(ctx context.Context, rawURL string) (*fetchResult, error) {
	bin := h.ytdlp
	if bin == "" {
		bin = "yt-dlp"
	}
	if _, err := exec.LookPath(bin); err != nil {
		return nil, fmt.Errorf("YouTube ingest needs yt-dlp on PATH (not found): %w", err)
	}
	tmp, err := os.MkdirTemp("", "chiton-yt-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmp)

	out := filepath.Join(tmp, "%(id)s.%(ext)s")
	// #nosec G204 -- bin is an operator-configured tool name; rawURL is an arg.
	cmd := exec.CommandContext(ctx, bin,
		"--skip-download",
		"--write-subs", "--write-auto-subs",
		"--sub-langs", "en.*",
		"--sub-format", "vtt",
		"--print", "after_move:title",
		"-o", out,
		rawURL,
	)
	titleBytes, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("yt-dlp failed: %w", err)
	}
	title := strings.TrimSpace(string(titleBytes))

	vtt, err := firstFileWithExt(tmp, ".vtt")
	if err != nil {
		return nil, fmt.Errorf("yt-dlp produced no subtitles for %s (none available?)", rawURL)
	}
	body, err := os.ReadFile(vtt)
	if err != nil {
		return nil, err
	}
	return &fetchResult{
		canonicalURL: rawURL,
		contentType:  "text/vtt",
		title:        title,
		ext:          "vtt",
		body:         body,
	}, nil
}

// firstFileWithExt returns the first file in dir whose name ends with ext.
func firstFileWithExt(dir, ext string) (string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	for _, e := range entries {
		if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ext) {
			return filepath.Join(dir, e.Name()), nil
		}
	}
	return "", fmt.Errorf("no %s file in %s", ext, dir)
}

// hostAllowed rejects hosts that are or resolve to loopback, private, link-local, or unspecified addresses
// (SSRF guard). DNS rebinding can still race this check and the dial. Closing that gap needs a custom dialer.
func hostAllowed(host string) error {
	if host == "" {
		return nil // file:// and other scheme-less inputs don't reach here
	}
	if ip := net.ParseIP(host); ip != nil {
		if isBlockedIP(ip) {
			return fmt.Errorf("refusing to fetch private/loopback address %s", host)
		}
		return nil
	}
	ips, err := net.LookupIP(host)
	if err != nil {
		return fmt.Errorf("resolve %s: %w", host, err)
	}
	for _, ip := range ips {
		if isBlockedIP(ip) {
			return fmt.Errorf("refusing to fetch %s — it resolves to private/loopback %s", host, ip)
		}
	}
	return nil
}

func isBlockedIP(ip net.IP) bool {
	return ip.IsLoopback() || ip.IsPrivate() || ip.IsUnspecified() ||
		ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast()
}

// requireRemote rejects anything but an absolute http(s) URL, so an assistant-driven add_url can't read local files.
func requireRemote(rawURL string) error {
	u, err := url.Parse(rawURL)
	if err != nil || u.Host == "" || (!strings.EqualFold(u.Scheme, "http") && !strings.EqualFold(u.Scheme, "https")) {
		return fmt.Errorf("add-url: only http/https URLs are accepted here (got %q); use the 'chiton add-url' CLI for local files", rawURL)
	}
	return nil
}

func isYouTube(host string) bool {
	host = strings.ToLower(host)
	host = strings.TrimPrefix(host, "www.")
	switch host {
	case "youtube.com", "m.youtube.com", "youtu.be", "music.youtube.com":
		return true
	}
	return false
}

// contentTypeForExt maps a file extension to a coarse MIME type for dispatch.
func contentTypeForExt(ext string) string {
	switch ext {
	case "pdf":
		return "application/pdf"
	case "html", "htm", "xhtml":
		return "text/html"
	case "md", "markdown":
		return "text/markdown"
	case "vtt":
		return "text/vtt"
	case "srt":
		return "application/x-subrip"
	case "json":
		return "application/json"
	default:
		return "text/plain"
	}
}

// inferKind guesses a content kind for the url node from the content type/host.
func inferKind(fr *fetchResult, rawURL string) string {
	if u, err := url.Parse(rawURL); err == nil && isYouTube(u.Host) {
		return "video"
	}
	switch {
	case strings.Contains(fr.contentType, "pdf"):
		return "paper"
	case strings.Contains(fr.contentType, "html"):
		return "article"
	default:
		return "document"
	}
}
