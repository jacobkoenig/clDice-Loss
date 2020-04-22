# clDice Loss Function  Keras/Tensorflow
Implementation of [clDice - a Novel Connectivity-Preserving Loss Function for Vessel Segmentation (2019)](https://arxiv.org/abs/2003.07311) in Keras/Tensorflow

Credit goes to [this repository](https://github.com/dmitrysarov/clDice/) which was used as a base for this implementation

---
### *clDice - a Novel Connectivity-Preserving Loss Function for Vessel Segmentation*
[[medneurips2019]](https://profs.etsmtl.ca/hlombaert/public/medneurips2019/27_CameraReadySubmission_cl_dice_neurips_med.pdf
)
[[arxiv]](https://arxiv.org/abs/2003.07311)
>Accurate segmentation of vascular structures is an emerging research topic with
>relevance to clinical and biological research. The connectedness of the segmented
>vessels is often the most significant property for many applications such as disease mo``deling for neurodegeneration and stroke. We introduce a novel metric
>namely clDice, which is calculated on the intersection of centerlines and volumes
>as opposed to the traditional dice, which is calculated on volumes only. Firstly,
>we tested state-of-the-art vessel segmentation networks using the proposed metric as evaluation criteria and show that it captures vascular network properties
>superior to traditional metrics, such as the dice-coefficient. Secondly, we propose
>a differentiable form of clDice as a loss function for vessel segmentation. We
>find that training on clDice leads to segmentation with more accurate connectivity
>information, higher graph similarity and often superior volumetric scores.

---
### Usage
`dice_helpers_tf.py` contains the conventional Dice loss function as well as clDice loss and its supplementary functions

Works with both image data formats `"channels_first"` and `"channels_last"`

##### Sample Usage
```python
from dice_helpers_tf import dice_loss, soft_cldice_loss

cldice_loss = soft_cldice_loss(k=5, data_format="channels_last")
model.compile(loss=cldice_loss, [...])

# Or combine dice + cldice similiar to the experiments in the paper
def combined_loss(y_true, y_pred):
    alpha = 0.5
    data_format="channels_last"
    return (alpha * dice_loss(data_format=data_format)(y_true, y_pred) + 
            (1-alpha) * soft_cldice_loss(k=5, data_format=data_format)(y_true, y_pred))
            
model.compile(loss= combined_loss, [...])

```

