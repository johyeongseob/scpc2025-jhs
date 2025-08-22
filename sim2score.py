import torch
import torch.nn.functional as F


def compute_similarity(outputs) -> torch.Tensor:
    """
    clip_outputs: output of CLIPModel(**inputs)
    return: clip_sim (torch.Tensor) of shape (1, 4) - similarity scores between image and texts
    """
    return torch.matmul(outputs.image_embeds, outputs.text_embeds.T)  # shape: (1, 4)


def similarity_to_score_dict(sim: torch.Tensor) -> dict:
    """
    sim: shape (1, 4) or (4,)
    return: dict like {'A': 4, 'B': 3, 'C': 2, 'D': 1}
    """
    labels = ["A", "B", "C", "D"]

    sim = sim.squeeze(0)

    sorted_indices = torch.argsort(sim, descending=True)

    rank_to_score = {rank: 4 - i for i, rank in enumerate(sorted_indices.tolist())}

    score_dict = {labels[i]: rank_to_score[i] for i in range(4)}

    return score_dict


def select_final_answer(clip_score_dict: dict, siglip_score_dict: dict) -> str:
    """
    clip_score_dict: e.g., {'A': 4, 'B': 2, 'C': 3, 'D': 1}
    siglip_score_dict: e.g., {'A': 2, 'B': 4, 'C': 3, 'D': 1}
    return: final_answer (str) → one of 'A', 'B', 'C', 'D'
    """
    labels = ["A", "B", "C", "D"]

    final_score = {
        k: clip_score_dict[k] + siglip_score_dict[k]
        for k in labels
    }

    max_score = max(final_score.values())
    candidates = [k for k, v in final_score.items() if v == max_score]

    if len(candidates) == 1:
        return candidates[0]
    else:
        return max(candidates, key=lambda k: clip_score_dict[k])


# def select_final_answer_sim(clip_sim: torch.Tensor, siglip_sim: torch.Tensor) -> str:
#     """
#     clip_sim, siglip_sim: shape (1, 4)
#     """
#     clip_sim = F.softmax(clip_sim.squeeze(0), dim=0)
#     siglip_sim = F.softmax(siglip_sim.squeeze(0), dim=0)
#
#     avg_sim = (clip_sim + siglip_sim) / 2
#     idx = torch.argmax(avg_sim).item()
#     return ["A", "B", "C", "D"][idx]
