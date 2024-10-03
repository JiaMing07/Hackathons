from ImageReward import ReFL_lora
from ImageReward.ReFL_lora import parse_args
if __name__ == "__main__":
    args = parse_args()
    trainer = ReFL_lora.Trainer("checkpoint/stable-diffusion-v1-4", "data/refl_data.json", args=args)
    trainer.train(args=args)
