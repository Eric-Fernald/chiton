package analyze

import (
	"context"

	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/walk"
)

// walkSources calls fn with each regular file under root that an extractor supports, under the shared ignore
// policy (.chitonignore included) and never through a symlink. Unreadable entries go to skipped; ctx ends it.
func walkSources(ctx context.Context, root string, fn func(path string) error) (skipped []string, err error) {
	policy, err := walk.LoadPolicy(root)
	if err != nil {
		return nil, err
	}
	_, err = walk.Files(ctx, root, policy, func(path string) error {
		if extast.ForFile(path) == nil {
			return nil
		}
		return fn(path)
	}, func(path string, _ error) { skipped = append(skipped, path) })
	return skipped, err
}
