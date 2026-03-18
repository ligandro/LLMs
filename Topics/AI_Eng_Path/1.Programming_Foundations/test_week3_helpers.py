
from week3_helpers import normalize_post, posts_to_user_summary

def test_normalize_post_trims_title():
    data = {"id": 1, "userId": 2, "title": "  hello  ", "body": "abc"}
    normalized = normalize_post(data)
    assert normalized["title"] == "hello"
    assert normalized["post_id"] == 1

def test_posts_to_user_summary_counts_posts():
    posts = [
        {"userId": 1, "id": 1, "title": "a", "body": "x"},
        {"userId": 1, "id": 2, "title": "b", "body": "y"},
        {"userId": 2, "id": 3, "title": "c", "body": "z"},
    ]
    summary = posts_to_user_summary(posts)
    assert summary[1] == 2
    assert summary[2] == 1
