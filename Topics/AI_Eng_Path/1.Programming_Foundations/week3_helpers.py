
def normalize_post(post: dict) -> dict:
    return {
        "post_id": post.get("id"),
        "user_id": post.get("userId"),
        "title": str(post.get("title", "")).strip(),
        "body_preview": str(post.get("body", ""))[:60],
    }

def posts_to_user_summary(posts: list[dict]) -> dict[int, int]:
    summary: dict[int, int] = {}
    for post in posts:
        user_id = int(post.get("userId", 0))
        summary[user_id] = summary.get(user_id, 0) + 1
    return summary
