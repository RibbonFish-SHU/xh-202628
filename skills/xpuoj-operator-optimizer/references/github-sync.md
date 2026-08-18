# Local GitHub Synchronization

GitHub operations belong on the local control machine. The NVIDIA server does not need `gh`, GitHub MCP, a deploy key, a PAT, or outbound GitHub access.

## Information Required From The User

- Existing repository clone URL; or, for a new repository, owner, name, and public/private visibility.
- Desired Git commit author name and email; a GitHub noreply email is acceptable.
- Default branch only if it should differ from `main`.

Do not ask the user to send a GitHub password, private SSH key, Cookie, or PAT in chat.

## Preferred Local Setup

GitHub has both an official CLI and an official MCP server. Prefer the official `gh` CLI for deterministic repository creation/authentication and normal `git` for commits and pushes. Use GitHub MCP only when the current Agent host has already configured it securely and its tool permissions are clear.

On Windows, install the official CLI with WinGet when it is absent:

```powershell
winget install --id GitHub.cli --source winget
```

For CLI authentication, use the one-time browser flow:

```powershell
gh auth login --web
gh auth status
```

The user completes GitHub authorization in the opened browser. Do not print authentication tokens. After authentication, an Agent can create and connect a repository without repeatedly operating the website, for example:

```powershell
gh repo create OWNER/REPO --private --source . --remote origin --push
```

Replace `--private` with `--public` only after the user explicitly chooses public visibility. For an existing repository, configure the exact confirmed URL as `origin`, review divergence, and never overwrite unrelated history.

## Repository Rules

- Configure repository-local `user.name` and `user.email`; do not change global Git config without permission.
- Run secret/privacy checks before the first push, especially for a public repository.
- Keep code, tests, small benchmark CSV/JSON, Skill files, and documentation in Git.
- Keep credentials, raw browser captures containing personal data, build trees, datasets, raw profiler output, and remote result dumps out of Git.
- Never force-push or rewrite commits used for remote tests, XPU-OJ submissions, or user reports.
- Record the exact GitHub URL and default branch in `state/PROJECT_STATE.md` after setup.

## Official References

- GitHub CLI manual: <https://cli.github.com/manual/>
- Official Windows installation: <https://github.com/cli/cli/blob/trunk/docs/install_windows.md>
- `gh auth login`: <https://cli.github.com/manual/gh_auth_login>
- `gh repo create`: <https://cli.github.com/manual/gh_repo_create>
- Official GitHub MCP Server: <https://github.com/github/github-mcp-server>
