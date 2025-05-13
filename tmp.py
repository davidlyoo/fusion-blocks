if lvl in [2, 3, 4]:
    self.fusion_blocks.append(
        DDCIBlock(
            in_channels=ch,
            num_heads=num_heades_per_level[lvl],
            spatial_reduction=16,
            channel_reduction=4
            )  
    )
else:
    self.fusion_blocks.append(
        SFBlock(
            channels=ch,
            num_heads=num_heads_per_level[lvl],
            dims=dims
            )
    )
