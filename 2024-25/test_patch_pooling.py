
import pytest
import torch


class PatchPooling(torch.nn.Module):
    def forward(self, batch: torch.Tensor, patch_lengths: torch.Tensor) -> torch.Tensor:
        B, S, D = batch.shape
        B_1, P = patch_lengths.shape

        assert B == B_1

        ### Your code goes here ###

        patch_ends = torch.cumsum(input = patch_lengths, dim = 1)
        patch_beginings = patch_ends-patch_lengths
        indexes = torch.arange(0, S)

        mask = (indexes.view((1, 1, -1)) < patch_ends.view((B, P, 1))) & \
            (indexes.view((1, 1, -1)) >= patch_beginings.view((B, P, 1)))

        patch_sum = torch.matmul(mask, batch)

        patch_lengths = torch.where(patch_lengths==0, 1, patch_lengths).unsqueeze(-1)
        safe_lengths = patch_lengths.clamp(min=1.0)
        result = patch_sum/safe_lengths
        result = torch.where(patch_lengths > 0, result, -1*torch.ones_like(result))

        return result

        ###########################




class TestPatchPooling:
    @pytest.mark.parametrize(
        "batch,patch_lengths,expected_output",
        [
            (
                torch.tensor(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0, 3.0, 3.0],
                            [3.0, 3.0, 3.0, 3.0, 3.0],
                        ],
                        [
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [5.0, 5.0, 5.0, 5.0, 5.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                        [
                            [6.0, 6.0, 6.0, 6.0, 6.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                    ]
                ),
                torch.tensor([[3, 1, 2], [4, 1, 0], [1, 0, 0]]),
                torch.tensor(
                    [
                        [
                            [1.0, 1.0, 1.0, 1.0, 1.0],
                            [2.0, 2.0, 2.0, 2.0, 2.0],
                            [3.0, 3.0, 3.0, 3.0, 3.0],
                        ],
                        [
                            [4.0, 4.0, 4.0, 4.0, 4.0],
                            [5.0, 5.0, 5.0, 5.0, 5.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                        [
                            [6.0, 6.0, 6.0, 6.0, 6.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                            [-1.0, -1.0, -1.0, -1.0, -1.0],
                        ],
                    ]
                ),
            )
        ],
    )
    def test_forward(
        self,
        batch: torch.Tensor,
        patch_lengths: torch.Tensor,
        expected_output: torch.Tensor,
    ) -> None:
        # given
        patch_pooling = PatchPooling()

        # when
        output = patch_pooling(batch=batch, patch_lengths=patch_lengths)

        # then
        assert torch.all(torch.isclose(output, expected_output))

