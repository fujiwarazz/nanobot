---
description: "Install a skill from a GitHub repo into the workspace skills directory."
metadata: {"nanobot":{"always":false}}
---

# Skill: skill-installer

Use this skill when the user asks to install/download a skill.

## What this skill does
- Uses the CLI to download a skill from a GitHub repo and install it into the workspace.

## How to use
1. Ask the user for:
   - Repo URL (or owner/repo)
   - Skill name (folder that contains `SKILL.md`)
   - Optional ref (branch/tag)
2. Run:

```bash
nanobot add <repo_or_url> --skill <skill_name> [--ref <branch_or_tag>]
```

## Example
```bash
nanobot add https://github.com/vercel-labs/skills --skill find-skills
```

## Notes
- If the skill already exists, rerun with `--force` to overwrite.
- If the repo is private, the user must have git credentials set up.
