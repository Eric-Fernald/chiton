// Package bearer holds the bearer-token GET shared by the Slack and Gmail REST clients.
package bearer

import (
	"context"
	"io"
	"net/http"
)

// maxBodyBytes caps how much of a response body Get reads.
const maxBodyBytes = 32 << 20

// Get sends an "Authorization: Bearer <token>" GET to reqURL, returning the status code and up to 32 MiB of
// the body (a read error truncates it). Callers check the status themselves.
func Get(ctx context.Context, client *http.Client, reqURL, token string) (status int, body []byte, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL, nil)
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Authorization", "Bearer "+token)
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	body, _ = io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	return resp.StatusCode, body, nil
}
