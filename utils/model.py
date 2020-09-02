from fastai.vision import *
import torch.nn as nn

class FastVision:
  def __init__(self, arch, parameters, task=None, arch_type="convnet"):
    self.arch = arch
    if arch_type.lower() == "convnet":
      self.__init_cnn(task, parameters)
    elif arch_type.lower() == "unet":
      self.__init_unet(parameters)
    else:
      raise NameError(f'This architecture "{arch_type}"" does not support! "convnet" or "unet"')

  def __init_cnn(self, task, parameters):
    self.task = task
    if task.lower() == "classification":
      self.__init_cnn_cls(parameters)
    elif task.lower() == "regression":
      self.__init_cnn_reg(parameters)
    elif task is None:
      raise ValueError(f'task Value is None! "classification" or "regression"')
    else:
      raise ValueError(f'This task "{task}"" does not support! "classification" or "regression"')

  def __init_cnn_cls(self, parameters):
    """
    Initialize CNN parameter for Classification
    Create DataBunch and Learner for Classification
    """
    path = parameters["path"]
    bs = parameters["bs"] 
    size = parameters["size"] 
    valid_pct = parameters["valid_pct"]

    metrics = parameters["metrics"]
    pretrained = parameters["pretrained"] 
    load_dir = parameters["load_dir"]


    bunch = self.create_cls_cnn_databunch(path, bs, size, valid_pct)
    bunch.show_batch(rows=3, figsize=(12,9))

    self.learn = self.create_cls_cnn_learner(bunch, metrics, pretrained, load_dir)

  def create_cls_cnn_databunch(self, path, bs=16, size=128, valid_pct=0.2):
    tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
    src = (ImageList.from_folder(path)
       .split_by_rand_pct(valid_pct)
       .label_from_folder())
    
    bunch = (src.transform(tfms, size=size)
        .databunch(bs=bs, no_check=True).normalize(imagenet_stats))
    return bunch 

  def create_cls_cnn_learner(self, data, metrics=[accuracy], pretrained=True, load_dir=None):
    learn = cnn_learner(data, self.arch, metrics=metrics, pretrained=pretrained)
    if load_dir is not None:
      learn.load(load_dir) 
    return learn

  def __init_cnn_reg(self, parameters):
    """
    Initialize CNN parameter for Regression
    Create DataBunch and Learner for Regression
    """

    df = parameters["df"]
    colnames = parameters["colnames"] 
    parent_path = parameters["parent_path"]
    folder = parameters["folder"]
    bs = parameters["bs"] 
    size=parameters["size"] 
    valid_pct=parameters["valid_pct"]

    metrics = parameters["metrics"]
    pretrained = parameters["pretrained"] 
    load_dir = parameters["load_dir"]

    final_size = parameters["final_size"]
    y_range = parameters["y_range"]

    bunch = self.create_reg_cnn_databunch(df, colnames, parent_path, folder,
                                          bs, size, valid_pct)
    bunch.show_batch(rows=3, figsize=(12,9))

    self.learn = self.create_reg_cnn_learner(bunch, final_size, y_range, metrics, pretrained, load_dir)

  def __init_unet(self, parametes):
    pass


  def create_reg_cnn_databunch(self, df, colnames, parent_path, folder, bs=16, size=128, valid_pct=0.2):

    label_df = pd.read_csv(parent_path / df)[pd.Index(colnames)]
    print(label_df.head())

    tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
    
    src = (ImageList.from_df(label_df,  parent_path, folder=folder)
       .split_by_rand_pct(0.2)
       .label_from_df(label_cls=FloatList))
 
    bunch = (src.transform(tfms, size=size)
             .databunch(bs=bs, no_check=True).normalize(imagenet_stats))
    return bunch 


  

  def create_reg_cnn_learner(self, data, final_size, y_range=None, metrics=[root_mean_squared_error, r2_score], pretrained=True, load_dir=None):

    model = ImageRegression(self.arch, final_size, pretrained, y_range)

    learn = Learner(data, model, metrics=metrics,
                    callback_fns=[ShowGraph]).mixup(stack_y=False, alpha=0.2)
    learn.loss_func = L1LossFlat()

    learn = cnn_learner(data, self.arch, metrics=metrics, pretrained=pretrained)
    if load_dir is not None:
      learn.load(load_dir) 
    return learn

  def find_alpha(self, suggestion=True):
    self.learn.lr_find()
    self.learn.recorder.plot(suggestion=suggestion)

  def fit_model_cyc(self, epoch, alpha=5e-02, wd=0.2):
    """
    fit_one_cycle로 모델을 학습 
    learner를 학습하고, 모델의 확률 예측함수 반환 
    """
    self.learn.model = nn.DataParallel(self.learn.model)
    self.learn.fit_one_cycle(epoch, alpha, wd=wd)
    self.predict_prob = lambda x: np.array([self.learn.predict(row)[2].detach().numpy().astype(float) for _, row in pd.DataFrame(x, columns=fastlime.feature_names).iterrows()])    



class ImageRegression(nn.Module):
  def __init__(self, arch, final_size, pretrained=True, y_range=None):
    super().__init__()
    layers = list(arch(pretrained=pretrained).children())[:-2]
    layers += [AdaptiveConcatPool2d(), Flatten()]
    layers += [nn.BatchNorm1d(final_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
    layers += [nn.Dropout(p=0.50)]
    layers += [nn.Linear(final_size, int(final_size / 2), bias=True), nn.ReLU(inplace=True)]
    layers += [nn.BatchNorm1d(int(final_size / 2), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)]
    layers += [nn.Dropout(p=0.50)]
    layers += [nn.Linear(int(final_size / 2), 16, bias=True), nn.ReLU(inplace=True)]
    layers += [nn.Linear(16,1)]
    self.imagereg = nn.Sequential(*layers)
    self.y_range = y_range
    
  def forward(self, x):
    x = self.imagereg(x).squeeze(-1)

    if self.y_range is not None:
      x = (self.y_range[1]-self.y_range[0]) * torch.sigmoid(x) + self.y_range[0]
    return x

class L1LossFlat(nn.SmoothL1Loss):
  def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
    return super().forward(input.view(-1), target.view(-1))