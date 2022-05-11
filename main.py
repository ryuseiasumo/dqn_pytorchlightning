import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="main.yaml")
def main(config: DictConfig):
    print("current directry:", config.original_work_dir) # /home0/hatar/self_study/dqn_pytorchlightning
    
    if config.do_train:
        # trainの処理
        from src.train import train
            
        # Train model
        train(config)
    
    
    


if __name__ == "__main__":
    #python main.py do_train=False #コマンドライン引数でhydraで与えた変数の値の変更も可能
    #python main.py --multirun model=dqn,mnist #いろんな条件でやりたい時？ 
    
    main()
