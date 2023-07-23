from ImageReward import ReFL_lora

if __name__ == "__main__":
    args = ReFL_lora.parse_args()
    trainer = ReFL_lora.Trainer("checkpoint/stable-diffusion-v1-4", "data/refl_data.json", args=args)
    trainer.train(args=args)
