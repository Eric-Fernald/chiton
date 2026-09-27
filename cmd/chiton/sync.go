package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	_ "github.com/Eric-Fernald/chiton/internal/analyze/rules"
	"github.com/Eric-Fernald/chiton/internal/embed"
	"github.com/Eric-Fernald/chiton/internal/ingest"
	gitpkg "github.com/Eric-Fernald/chiton/internal/ingest/git"
)

// syncCmd is the parent for chiton's live-sync front-ends (Phase 3).
func syncCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "sync",
		Short: "Keep the graph in step with the source as files change",
		Long: `Incrementally update the graph when files change, instead of a full rebuild.

  chiton sync watch [path]   watch a tree with fsnotify and re-index on save
  chiton sync git            re-index files changed by git (post-commit hook)
  chiton sync once [path]    one-shot incremental reconcile (re-index + prune)

Each changed file is re-extracted, its stale nodes/edges replaced, and (when
enabled) its embeddings and deterministic findings refreshed. After a batch
settles, the touched 2-hop neighborhood is incrementally re-clustered, updating
chiton-out/cluster.json without a full pass (run 'chiton cluster' once to seed
it). Install the git hooks with 'chiton hooks install --stage post-commit,post-checkout'.`,
	}
	cmd.AddCommand(syncWatchCmd(), syncGitCmd(), syncOnceCmd())
	return cmd
}

// syncEngineFlags are the engine-tuning flags shared by the sync subcommands.
type syncEngineFlags struct {
	reembed    bool
	reanalyze  bool
	resolution float64
	workers    int
	debounceMS int
	clusterMS  int
	provider   string
	model      string
}

func (f *syncEngineFlags) bind(cmd *cobra.Command, watch bool) {
	cmd.Flags().BoolVar(&f.reembed, "embed", false, "Re-embed changed files' nodes (requires an embedding provider)")
	cmd.Flags().BoolVar(&f.reanalyze, "analyze", true, "Re-run deterministic file rules on changed files")
	cmd.Flags().Float64Var(&f.resolution, "resolution", 1.0, "Leiden resolution for incremental re-clustering")
	cmd.Flags().StringVar(&f.provider, "provider", "", "Embedding provider override for --embed (ollama|hash)")
	cmd.Flags().StringVar(&f.model, "model", "", "Embedding model override for --embed")
	if watch {
		cmd.Flags().IntVar(&f.workers, "workers", 4, "Concurrent re-extraction workers")
		cmd.Flags().IntVar(&f.debounceMS, "debounce-ms", 300, "Per-file debounce window in milliseconds")
		cmd.Flags().IntVar(&f.clusterMS, "cluster-debounce-ms", 3000, "Idle window before an incremental re-cluster, in milliseconds")
	}
}

// openSyncEngine opens the store read-write, loads the cache, and wires the
// optional embedder/analyzer per flags. The returned closer must be called.
func openSyncEngine(cmd *cobra.Command, f *syncEngineFlags) (*ingest.Engine, func() error, error) {
	cfg, gs, err := loadStore(false)
	if err != nil {
		return nil, nil, err
	}
	ch, err := loadBuildCache(cfg.GraphPath)
	if err != nil {
		gs.Close()
		return nil, nil, err
	}

	var embedder embed.Embedder
	if f.reembed {
		embedder, err = newEmbedder(cfg, f.provider, f.model)
		if err != nil {
			fmt.Fprintf(cmd.ErrOrStderr(), "sync: warning: --embed disabled: %v\n", err)
			embedder = nil
		}
	}

	var analyzer *analyze.Analyzer
	if f.reanalyze {
		analyzer = analyze.New(gs, nil)
	}

	eng := ingest.New(gs, ch, ingest.Config{
		GraphPath:       cfg.GraphPath,
		Resolution:      f.resolution,
		Reembed:         embedder != nil,
		Reanalyze:       analyzer != nil,
		Workers:         f.workers,
		Debounce:        time.Duration(f.debounceMS) * time.Millisecond,
		ClusterDebounce: time.Duration(f.clusterMS) * time.Millisecond,
	}, embedder, analyzer, errLogf(cmd))

	return eng, gs.Close, nil
}

func syncWatchCmd() *cobra.Command {
	var f syncEngineFlags
	cmd := &cobra.Command{
		Use:   "watch [path]",
		Short: "Watch a directory tree and re-index files as they change",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			root := "."
			if len(args) == 1 {
				root = args[0]
			}
			eng, closeStore, err := openSyncEngine(cmd, &f)
			if err != nil {
				return err
			}
			defer closeStore()

			// cmd.Context() is cancelled by the root signal handler on
			// Ctrl-C / SIGTERM, so the watcher drains and flushes.
			fmt.Fprintf(cmd.OutOrStdout(), "sync: watching %s (re-embed=%v re-analyze=%v)\n", root, f.reembed, f.reanalyze)
			return eng.Watch(cmd.Context(), root)
		},
	}
	f.bind(cmd, true)
	return cmd
}

func syncGitCmd() *cobra.Command {
	var f syncEngineFlags
	var root, since, stage string
	cmd := &cobra.Command{
		Use:   "git [-- hook-args...]",
		Short: "Re-index files changed by git (drives the post-commit/post-checkout hooks)",
		Long: `Resolve the set of files git changed and incrementally re-index them.

  --stage post-commit     re-index the files in the just-made commit (HEAD)
  --stage post-checkout   re-index files differing between two checked-out trees
                          (git passes <old> <new> <flag> as positional args)
  --since <ref>           re-index files changed between <ref> and HEAD

With no flag, defaults to the files in HEAD (post-commit semantics).`,
		RunE: func(cmd *cobra.Command, args []string) error {
			if !gitpkg.Available(root) {
				return fmt.Errorf("sync git: %s is not a git work tree (or git is not on PATH)", root)
			}
			changes, skip, err := resolveGitChanges(root, since, stage, args)
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			if skip {
				fmt.Fprintln(out, "sync git: nothing to do (file checkout, not a branch switch)")
				return nil
			}
			if len(changes) == 0 {
				fmt.Fprintln(out, "sync git: no changed files")
				return nil
			}

			eng, closeStore, err := openSyncEngine(cmd, &f)
			if err != nil {
				return err
			}
			defer closeStore()

			sum, err := eng.SyncChanges(cmd.Context(), ingest.GitChanges(changes))
			if err != nil {
				return err
			}
			fmt.Fprintf(out, "sync git: %d upserted, %d deleted, %d skipped, %d re-embedded%s\n",
				sum.Upserted, sum.Deleted, sum.Skipped, sum.Reembedded, reclusterNote(sum.Reclustered))
			return nil
		},
	}
	f.bind(cmd, false)
	cmd.Flags().StringVar(&root, "root", ".", "Repository root")
	cmd.Flags().StringVar(&since, "since", "", "Re-index files changed since this git ref")
	cmd.Flags().StringVar(&stage, "stage", "", "Hook stage: post-commit | post-checkout")
	return cmd
}

// resolveGitChanges turns the stage/since/args into a change set. skip is true
// when a post-checkout was a file checkout (no branch switch), which is a no-op.
func resolveGitChanges(root, since, stage string, args []string) (changes []gitpkg.Change, skip bool, err error) {
	switch {
	case since != "":
		changes, err = gitpkg.ChangedSince(root, since)
	case stage == "post-checkout":
		// git post-checkout args: <prev-HEAD> <new-HEAD> <branch-flag>.
		if len(args) >= 3 && args[2] == "0" {
			return nil, true, nil // file checkout, not a branch switch
		}
		// After a clone or `git worktree add`, prev-HEAD is the null SHA, which `git diff`
		// rejects ("bad object"). Index the new HEAD instead.
		if len(args) >= 2 && !isNullSHA(args[0]) {
			changes, err = gitpkg.RangeChanges(root, args[0], args[1])
		} else {
			changes, err = gitpkg.LastCommitChanges(root)
		}
	default: // post-commit or unspecified
		changes, err = gitpkg.LastCommitChanges(root)
	}
	return changes, false, err
}

// isNullSHA reports whether s is git's all-zero object id (passed as prev-HEAD
// by the first post-checkout after a clone or worktree add).
func isNullSHA(s string) bool {
	if s == "" {
		return true
	}
	return strings.Trim(s, "0") == ""
}

func syncOnceCmd() *cobra.Command {
	var f syncEngineFlags
	var prune bool
	cmd := &cobra.Command{
		Use:   "once [path]",
		Short: "One-shot incremental reconcile of a directory (re-index changed, prune removed)",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			root := "."
			if len(args) == 1 {
				root = args[0]
			}
			eng, closeStore, err := openSyncEngine(cmd, &f)
			if err != nil {
				return err
			}
			defer closeStore()

			sum, err := eng.SyncDir(cmd.Context(), root, prune)
			if err != nil {
				return err
			}
			fmt.Fprintf(cmd.OutOrStdout(), "sync once: %d upserted, %d deleted, %d skipped, %d re-embedded%s\n",
				sum.Upserted, sum.Deleted, sum.Skipped, sum.Reembedded, reclusterNote(sum.Reclustered))
			return nil
		},
	}
	f.bind(cmd, false)
	cmd.Flags().BoolVar(&prune, "prune", true, "Delete nodes for files under the tree that no longer exist")
	return cmd
}

func reclusterNote(reclustered bool) string {
	if reclustered {
		return " (re-clustered)"
	}
	return ""
}
