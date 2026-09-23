# GitHub Actions workflows

This directory holds every workflow for the repository. There are no
subdirectories: GitHub only discovers workflows placed directly under
`workflows`, so reusable workflows live here too, alongside the
top-level ones that trigger on events.

## Naming convention

Every file is prefixed by what it does:

| Prefix      | Purpose                                                                                                                                                              |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ci-`       | Quality/test checks, most of them reusable workflows called from `workflows/ci.yaml`.                                                                                |
| `lib-`      | Reusable workflows with no event trigger of their own; they only expose `workflow_call` outputs (e.g. reading a JSON config file) and are `needs:`-ed by other jobs. |
| `bot-`      | Scheduled automation that pushes commits / opens PRs as the `ci-bot` GitHub App.                                                                                     |
| `nightly-`  | Scheduled checks against the _published_ PyPI package (as opposed to `ci-*`, which checks the repo's source).                                                        |
| `release-`  | Publishing: PyPI package, GitHub release assets, documentation.                                                                                                      |
| `security-` | Supply-chain/security scanning (Scorecard, dependency review).                                                                                                       |

`workflows/ci.yaml` is the entry point for pull requests/pushes to `main`; it fans out
to the `ci-*.yaml` reusable workflows so each check can also be run/dispatched
on its own. Job ids in `ci.yaml` and in the workflows it calls are
load-bearing for branch protection (see the comment at the top of `ci.yaml`)
— keep them in sync if you rename either side.

`ci-verify-workflows.yaml` is deliberately _not_ called from `ci.yaml`: it
lints `.github/workflows/*` and `.github/actions/*` itself, so it triggers on
changes to those paths rather than every PR/push.

## Composite actions vs. reusable workflows

- **`.github/actions/*`** (composite actions): used when the shared unit is a
  handful of _steps_ inside a single job (e.g. configuring git and fetching
  `gh-pages` for `mike`, or verifying an already-installed package). Composite
  actions cannot produce a job-level output usable in a `strategy.matrix`.
  Currently: `setup-doc-deploy` (shared by `ci-doctest.yaml` and
  `release-docs-publish.yaml`) and `verify-installed-package` (shared by the
  `nightly-test-package*.yaml` workflows).
- **`.github/workflows/lib-*.yaml`** (reusable workflows called with `uses:
./.github/workflows/lib-....yaml`): used when the output needs to feed a
  matrix, or when the shared logic is naturally a whole job (e.g. loading
  `durandtibo/workflow-config-action`'s shared Python-version/OS config once
  and handing the JSON down to several jobs via `needs:`). Currently:
  `lib-get-test-matrix.yaml`,
  `lib-get-package-versions.yaml`, `lib-get-package-extras.yaml` and
  `lib-get-package-deps.yaml` (which combines the last two).

When adding new duplication, prefer extending an existing composite
action/reusable workflow over copy-pasting steps.

## Conventions applied to every workflow

- **Action pinning**: every third-party (and first-party `actions/*`) step is
  pinned to a full commit SHA with a trailing `# ratchet:owner/repo@vX.Y.Z`
  comment, added and refreshed automatically by
  [`ratchet`](https://github.com/sethvargo/ratchet) via `workflows/bot-pin-action.yaml`.
  Never hand-pin a SHA without that comment — the next `bot-pin-action` run
  would otherwise silently downgrade or drift it. Local composite actions
  (`uses: ./.github/actions/...`) are referenced by path, not pinned. Any
  runtime tool invoked outside of a pinned action (e.g. `npx <pkg>`) should
  also pin an explicit version, so a bot-authored PR can't pick up an
  unreviewed upstream release.
- **Permissions**: top-level `permissions:` is always the least the workflow
  needs — `contents: read` or `{}` — and any job that needs more (e.g.
  `contents: write` to push, `id-token: write` for OIDC) declares it on that
  job only, never widening the workflow default.
- **Timeouts**: every job sets `timeout-minutes`, sized to what the job
  actually does (2 for a config-read job, 5 for most checks, 10 for slower
  jobs like Scorecard) so a hung step can't occupy a runner indefinitely.
- **Runners**: `ubuntu-slim` for lightweight jobs that only run a small
  action or a couple of shell commands (no Python/build tooling); otherwise
  `ubuntu-latest`, or the OS matrix under test for `workflows/ci-test.yaml` /
  `nightly-test-package.yaml`.
- **`workflow_dispatch`**: added to every workflow (checks and automation
  alike) so it can be re-run manually without waiting for its normal trigger.

## Shared configuration

Supported Python versions and OS matrices are no longer duplicated in this
repo at all: `lib-get-test-matrix.yaml` loads them from
[`durandtibo/workflow-config-action`](https://github.com/durandtibo/workflow-config-action)'s
built-in default config, so that single upstream action is the one place to
update the version/OS lists across every repo that uses it.

Values that are still specific to this repo are centralized in
`../dev/config`:

- `../dev/config/package_versions.json` — version ranges to test both required
  and optional dependencies against (e.g. `coola`, `langchain-core`,
  `langchain-anthropic`, `polars`, ...), read by `lib-get-package-versions.yaml`
  and kept current by `bot-generate-package-versions.yaml`.

Optional-dependency _names_ used for build/install matrices (`ci-build.yaml`,
`nightly-test-package-extras.yaml`) are never duplicated in config: they're
read directly from `../pyproject.toml`'s `[project.optional-dependencies]` by
the `durandtibo/extract-pyproject-metadata-action` composite action /
`lib-get-package-extras.yaml`, so a new extra only needs to be added in one
place. `ci-test-deps.yaml` and `nightly-package.yaml` still list each
dependency (core and optional) as its own job, since most of the packages
tested there (`langchain-core`, `pydantic`, `rich`, ...) are required
dependencies rather than optional extras and so don't map onto
`pyproject.toml`'s extras list.

## Validating changes to this directory

`workflows/ci-verify-workflows.yaml` runs `durandtibo/verify-github-workflow-action` on
every PR/push that touches `.github/workflows/*` or `.github/actions/*`, checking pinning and
general workflow correctness. Locally, `actionlint` (run from this
directory) and a YAML syntax check are good pre-flight checks before pushing:

```bash
cd .github/workflows && actionlint
python3 -c "import yaml, glob; [yaml.safe_load(open(f)) for f in glob.glob('*.yaml') + glob.glob('*.yml')]"
```
