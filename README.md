## Reward Feedback Learning (ReFL) for LDMs

> Jiazheng Xu, 2023/07/23

### 代码准备

从/data/jiazheng/ImageReward-Hackathon复制到自己的工作文件夹

### 模型准备

在checkpoint/下复制粘贴以下ckpt：

```
bert-base-uncased/
stable-diffusion-v1-4/
ImageReward.pt
med_config.json
```

分别在：

```
/data/jiazheng/ImageReward-Hackathon/checkpoint/bert-base-uncased/
/data/jiazheng/ImageReward-Hackathon/checkpoint/stable-diffusion-v1-4/
/data/jiazheng/ImageReward-Hackathon/checkpoint/ImageReward.pt
/data/jiazheng/ImageReward-Hackathon/checkpoint/med_config.json
```

### 环境配置

```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_refl.txt
pip install tensorboard
```

### 伪代码

![WechatIMG4473 copy.png](https://s2.loli.net/2023/07/23/fvzsHeh2r8ZDSC1.png)

### 参考代码

```shell
ImageReward-Hackathon/reference/pipeline_stable_diffusion.py: L665~L779
ImageReward-Hackathon/reference/scheduling_ddpm.py: L384~L452
```

### 待填补代码

* 搜索：TODO

```shell
ImageReward-Hackathon/ImageReward/ReFL_lora.py
```

* 结果保存在：ImageReward-Hackathon/checkpoint/refl_lora

* 定性观察ReFL后的结果：

  ```bash
  python inference_lora.py 
  ```

  * 和ReFL之前做对比：

    ```bash
    python inference.py
    ```

* 调参：train_refl_lora.sh

  ```shell
  --learning_rate=1e-04 \
  --grad_scale 0.01 \
  ```

### 注意事项

1. 为了和他人共用机器，需要错开使用不同的GPU，例如在`ReFL.py`的`import os`后紧跟一行`os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'`（其中6,7可以是0-7中任意2个数，代表GPU编号，注意和他人不冲突）。本次ReFL需要使用2张卡训练。
2. 部分同学如果直接运行脚本可能会遇到端口号被占的报错，这是因为Accelerate默认端口号已经被其他同学占用，可以查阅`https://huggingface.co/docs/accelerate/package_reference/cli`官网，具体来说，【在`train_refl_lora.sh`第一行`accelerate launch`加上`--main_process_port 新的端口号`】即可。如果想要加深理解，可以自行Google，其定义为The port to use to communicate with the machine of rank 0.
3. 有同学遇到设置了os.environ['CUDA_VISIBLE_DEVICES']还是使用默认卡的情况，这可能是因为在os设置卡号之前Accelerate就使用到了GPU（可能是其内部实现用了什么feature），这时最简单的方法之一就是在代码最开始的地方，也即refl_lora.py的第一行import os并设置os.environ['CUDA_VISIBLE_DEVICES']。
4. inference的之后爆显存可能是因为默认文件没有给大家关闭梯度，手动加一行with torch.no_grad():即可。【小知识：通常来说占显存的大头往往是梯度】
5. 如果依然爆显存，可以把num_images_per_prompt从10改成1；这相当于缩小batch_size，也会大幅降低计算量

### 知识补充：
有同学问DDPM是啥，这来源于一篇paper：Denoising Diffusion Probabilistic Model(DDPM)，我们用的diffusers的reference中scheduling_ddpm代码就是DDPM作为schedule。
扩展阅读可以搜索DDPM+知乎：https://zhuanlan.zhihu.com/p/384144179；https://zhuanlan.zhihu.com/p/563661713
有个博客讲Diffusion也不错：https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

