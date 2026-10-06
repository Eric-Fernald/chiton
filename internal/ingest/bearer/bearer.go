// Package bearer holds the bearer-token GET shared by the Slack and Gmail REST clients.
package bearer

import (
	"context"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// maxBodyBytes caps how much of a response body Get reads.
const maxBodyBytes = 32 << 20

// maxRetries bounds how many times Get retries a 429 before returning it to the caller.
const maxRetries = 3

// maxRetryWait caps one wait, so a confused or hostile server can't stall an ingest.
const maxRetryWait = time.Minute

// sleep waits d or until ctx ends. A var so tests skip real waits.
var sleep = func(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// Get sends an "Authorization: Bearer <token>" GET to reqURL, returning the status code and up to 32 MiB of the
// body (a read error truncates it). It retries a 429 up to maxRetries times; callers check the final status.
func Get(ctx context.Context, client *http.Client, reqURL, token string) (status int, body []byte, err error) {
	for attempt := 0; ; attempt++ {
		var wait string
		status, body, wait, err = get(ctx, client, reqURL, token)
		if err != nil || status != http.StatusTooManyRequests || attempt == maxRetries {
			return status, body, err
		}
		if err := sleep(ctx, retryDelay(wait, attempt)); err != nil {
			return status, body, err
		}
	}
}

// get performs one request and also returns the response's Retry-After header.
func get(ctx context.Context, client *http.Client, reqURL, token string) (int, []byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return 0, nil, "", err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	return resp.StatusCode, body, resp.Header.Get("Retry-After"), nil
}

// retryDelay is the wait before retry attempt+1: Retry-After (seconds or an HTTP date) when present, else backoff
// doubling from 1s, capped at maxRetryWait.
func retryDelay(header string, attempt int) time.Duration {
	d := time.Second << attempt
	if secs, err := strconv.Atoi(strings.TrimSpace(header)); err == nil && secs >= 0 {
		d = time.Duration(secs) * time.Second
	} else if t, err := http.ParseTime(header); err == nil {
		d = time.Until(t)
	}
	return max(0, min(d, maxRetryWait))
}
