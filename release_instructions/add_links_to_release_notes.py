import re
from pathlib import Path


def linkify_pull_requests(text: str) -> str:
    """
    Replace instances of #<number> with markdown links to the corresponding
    AutoGluon GitHub pull request.
    """
    # Pattern matches # followed by 3 to 5 digits, ensures not already inside a markdown link
    pattern = r'(?<!\[)#(\d{3,5})(?!\])'

    def replacer(match):
        pr_number = match.group(1)
        return f"[#{pr_number}](https://github.com/autogluon/autogluon/pull/{pr_number})"

    return re.sub(pattern, replacer, text)


def linkify_user_mentions(text: str) -> str:
    """
    Replace instances of @username with markdown links to the corresponding
    GitHub user profile, unless already inside a markdown link.
    """
    # Match @username not already inside a markdown link
    pattern = r'(?<!\[)@([A-Za-z0-9-]+)(?!\])'

    def replacer(match):
        username = match.group(1)
        return f"[@{username}](https://github.com/{username})"

    return re.sub(pattern, replacer, text)


def unlinkify_user_mentions(text: str) -> str:
    """
    Revert GitHub user profile markdown links back to plain @username mentions.
    Example: [@Innixma](https://github.com/Innixma) -> @Innixma
    """
    pattern = r'\[@([A-Za-z0-9-]+)\]\(https://github\.com/\1\)'
    return re.sub(pattern, r'@\1', text)


def unlinkify_pull_requests(text: str) -> str:
    """
    Reverts GitHub pull request markdown links back to plain #1234 format.
    Example: [#5020](https://github.com/autogluon/autogluon/pull/5020) -> #5020
    """
    pattern = r'\[#(\d{3,5})\]\(https://github\.com/autogluon/autogluon/pull/\1\)'
    return re.sub(pattern, r'#\1', text)


def transform_changelog(file_path: str, strip_links_for_github_release: bool = False) -> None:
    """
    Reads a text file, applies GitHub pull request and user profile link transformations,
    and saves the updated text back to the original file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding='utf-8')

    text_w_pr_urls = linkify_pull_requests(text)
    text_w_user_urls = linkify_user_mentions(text_w_pr_urls)

    path.write_text(text_w_user_urls, encoding='utf-8')
    print(f"Updated file saved: {file_path}")

    if strip_links_for_github_release:
        path_stem = Path(file_path).stem
        path_suffix = Path(file_path).suffix
        path_wo_urls = Path(file_path).parent / f"{path_stem}_paste_to_github{path_suffix}"
        text_wo_pr_urls = unlinkify_pull_requests(text_w_user_urls)
        text_wo_user_urls = unlinkify_user_mentions(text_wo_pr_urls)
        path_wo_urls.write_text(text_wo_user_urls)
        print(f"Saved file for GitHub release notes pasting: {path_wo_urls}")


if __name__ == '__main__':
    """
    Run this to add urls for all pull requests and GitHub users in the `whats_new` markdown files.
    Uncomment the files you wish to update.
    
    Use the `vX.Y.Z_paste_to_github.md` file generating from running this script to paste the release notes into GitHub.
    This ensures that the URL links for PRs and GitHub users are removed,
    as having them breaks GitHub's contributor detection logic, and they are automatically added by GitHub.
    """
    file_prefix = "../docs/whats_new/"

    files = [
        # "v0.4.0.md",
        # "v0.4.1.md",
        # "v0.4.2.md",
        # "v0.4.3.md",
        # "v0.5.1.md",
        # "v0.5.2.md",
        # "v0.6.0.md",
        # "v0.6.1.md",
        # "v0.6.2.md",
        # "v0.7.0.md",
        # "v0.8.0.md",
        # "v0.8.1.md",
        # "v0.8.2.md",
        # "v0.8.3.md",
        # "v1.0.0.md",
        # "v1.1.0.md",
        # "v1.1.1.md",
        # "v1.2.0.md",
        # "v1.3.0.md",
    ]

    files = [file_prefix + f for f in files]

    for f in files:
        transform_changelog(f, strip_links_for_github_release=True)
