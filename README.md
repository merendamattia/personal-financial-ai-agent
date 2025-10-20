# merendamattia/devops-automation-hub

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Latest Release](https://img.shields.io/github/v/release/merendamattia/devops-automation-hub?label=release)](https://github.com/merendamattia/devops-automation-hub/releases)
[![Actions Status](https://github.com/merendamattia/devops-automation-hub/actions/workflows/check-docker-image.yaml/badge.svg)](https://github.com/merendamattia/devops-automation-hub/actions)
[![Actions Status](https://github.com/merendamattia/devops-automation-hub/actions/workflows/check-latex-document.yaml/badge.svg)](https://github.com/merendamattia/devops-automation-hub/actions)
[![Actions Status](https://github.com/merendamattia/devops-automation-hub/actions/workflows/conventional-commits-check.yaml/badge.svg)](https://github.com/merendamattia/devops-automation-hub/actions)
[![Actions Status](https://github.com/merendamattia/devops-automation-hub/actions/workflows/semantic-release.yaml/badge.svg)](https://github.com/merendamattia/devops-automation-hub/actions)
[![Actions Status](https://github.com/merendamattia/devops-automation-hub/actions/workflows/docker-release.yaml/badge.svg)](https://github.com/merendamattia/devops-automation-hub/actions)

A customizable GitHub Actions setup to streamline and automate development workflows.

Full environment setup:
```bash
pip install -r requirements.txt
```

## Actions supported
1. Git Conventional Commits check using [pre-commit](https://pre-commit.com/).
2. [Git Semantic Release](https://dev.to/sahanonp/how-to-setup-semantic-release-with-github-actions-31f3) using [action-for-semantic-release](https://github.com/marketplace/actions/action-for-semantic-release).
3. Auto Assign Pull Request by [kentaro-m/auto-assign-action](https://github.com/kentaro-m/auto-assign-action/tree/v2.0.0/).
4. Check Docker Image building.
5. Check LaTeX document building by [xu-cheng/latex-action](https://github.com/xu-cheng/latex-action/tree/v3/).
6. Build and push docker image to Docker Hub.

## Conventional Commits Hooks setup

This project follows the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. Please install the Git commit hooks before making commits:

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

## Auto Assign Pull Request setup

Follow these two steps to enable automatic assignment and allow Actions to create and approve pull requests.

1) Create a new ruleset in your repository: `Settings > Rules > Rulesets` and import the JSON below (do not modify the JSON). To make the file easier to read, the JSON is collapsed by default — click to expand:

<details>
<summary>Show JSON ruleset (click to expand)</summary>

```json
{
  "id": 5813723,
  "name": "Protect Main",
  "target": "branch",
  "source_type": "Repository",
  "source": "merendamattia/devops-automation-hub",
  "enforcement": "active",
  "conditions": {
    "ref_name": {
      "exclude": [],
      "include": [
        "~DEFAULT_BRANCH"
      ]
    }
  },
  "rules": [
    {
      "type": "deletion"
    },
    {
      "type": "non_fast_forward"
    },
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": false,
        "require_code_owner_review": true,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true,
        "automatic_copilot_code_review_enabled": true,
        "allowed_merge_methods": [
          "merge",
          "rebase"
        ]
      }
    },
    {
      "type": "copilot_code_review",
      "parameters": {
        "review_on_push": true,
        "review_draft_pull_requests": false
      }
    }
  ],
  "bypass_actors": [
    {
      "actor_id": 5,
      "actor_type": "RepositoryRole",
      "bypass_mode": "always"
    }
  ]
}
```

</details>

2) Grant workflows permission to create and approve PRs: go to `Settings > Actions > General > Workflow permissions`, set *"Read and write permissions"*, and enable *"Allow GitHub Actions to create and approve pull requests"*.

## Semantic Release setup
To use semantic release, create a GitHub token (`GH_TOKEN`) with repo permissions and add it as a repository secret named `GH_TOKEN`.

## Docker Release setup
To auto-build and push the Docker image after Semantic Release, add repository secrets:
- `DOCKERHUB_USERNAME`: your docker username;
- `DOCKERHUB_TOKEN`: your personal access token (PAT);
- `DOCKERHUB_REPO`: the Docker Hub repository (format: username/repo).


## Contributors
<a href="https://github.com/merendamattia/devops-automation-hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=merendamattia/devops-automation-hub" />
</a>
