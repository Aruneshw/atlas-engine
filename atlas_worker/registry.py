"""Algorithm registry: which atlas entries the engine can really train, and how.

Pure Python with no heavy imports, because both the GUI and the worker load it.
Entries are keyed by atlas entry number. Anything not listed here stays in the catalog
as reference and is labelled with why it cannot be trained (see `status_of`).

Class paths below are the only code the engine will ever import to build a model:
nothing typed in the UI reaches importlib or a shell.
"""
from __future__ import annotations

import re

TASKS = {
    "classification": "Classification",
    "regression": "Regression",
    "forecasting": "Time-series forecasting",
    "seq_classification": "Sequence classification",
    "anomaly": "Anomaly detection",
    "clustering": "Clustering",
    "dimreduction": "Dimensionality reduction",
    "image_classification": "Image classification",
    "object_detection": "Object detection",
    "semantic_segmentation": "Image segmentation",
    "representation": "Self-supervised embeddings",
    "image_generation": "Image generation",
    "audio_classification": "Audio classification",
    "video_classification": "Video classification",
    "text_classification": "Text classification",
    "text_generation": "LLM fine-tuning",
    "recommendation": "Recommendation",
    "node_classification": "Graph node classification",
    "reinforcement_learning": "Reinforcement learning",
}

ENVIRONMENTS = {
    "CartPole-v1": "Balance a pole on a cart (discrete actions)",
    "Acrobot-v1": "Swing a two-link arm upward (discrete actions)",
    "MountainCar-v0": "Drive a car up a hill (discrete actions)",
    "Pendulum-v1": "Hold a pendulum upright (continuous actions)",
    "MountainCarContinuous-v0": "Drive a car up a hill (continuous actions)",
    "FrozenLake-v1": "Cross a frozen lake (grid, tabular methods)",
    "CliffWalking-v1": "Walk along a cliff (grid, tabular methods)",
    "Taxi-v3": "Pick up and drop off passengers (grid, tabular methods)",
    "Bandit-10": "Ten slot machines with hidden payouts (bandit methods)",
}
NEEDS = {"forecaster": "steps", "seq_forecast": "steps", "image": "images", "detect_tv": "images", "detect_ultra": "images", "segment": "images", "selfsup": "images",
         "audio": "audio", "video": "video", "gen": "samples", "rl_sb3": "episodes", "rl_table": "episodes", "bandit": "episodes", "llm": "prompt", "graph": "graph"}

# Which runtime package groups exist. Pinned versions live in atlas_worker/runtime_pins.py.
GROUPS = ("core", "formats", "boost", "catboost", "stats", "pyod", "umap", "torch", "vision", "text", "peft", "timm", "ultra", "smp", "audio", "video", "rl", "graph")


# ---- parameter helpers -------------------------------------------------------------
def I(name, default, lo, hi, tune=None, to="model", label=None):
    return dict(name=name, type="int", default=default, min=lo, max=hi, tune=tune, to=to, label=label or name)


def F(name, default, lo, hi, tune=None, to="model", label=None):
    return dict(name=name, type="float", default=default, min=lo, max=hi, tune=tune, to=to, label=label or name)


def C(name, default, choices, tune=None, to="model", label=None):
    return dict(name=name, type="choice", default=default, choices=list(choices), tune=tune, to=to, label=label or name)


def TXT(name, default, label=None):
    """A short identifier such as a Hugging Face model name. Checked against a strict pattern."""
    return dict(name=name, type="text", default=default, to="model", label=label or name, tune=None)


def OI(name, default, lo, hi, tune=None, to="model", label=None):
    """Optional int: 0 in the UI means None."""
    return dict(name=name, type="optional_int", default=default, min=lo, max=hi, tune=tune, to=to, label=label or name)


ALPHA = F("alpha", 1.0, 1e-6, 1e4, [0.01, 0.1, 1, 10, 100])
TREES = I("n_estimators", 200, 10, 3000, [100, 200, 400])
DEPTH = OI("max_depth", 0, 0, 100, [0, 6, 12, 24], label="max_depth (0 = unlimited)")
LR = F("learning_rate", 0.1, 1e-4, 1.0, [0.03, 0.1, 0.3])
KNN = I("n_neighbors", 5, 1, 200, [3, 5, 11, 21])
SVM_C = F("C", 1.0, 1e-4, 1e4, [0.1, 1, 10])
KERNEL = C("kernel", "rbf", ["rbf", "linear", "poly", "sigmoid"], ["rbf", "linear"])
CLUSTERS = I("n_clusters", 3, 2, 200)
COMPONENTS = I("n_components", 2, 1, 200)
SEED = {"random_state": 42}


# ---- impl helpers ------------------------------------------------------------------
def sk(path, **kw):
    return dict(adapter="sk", cls=path, **kw)


def obj(path, **kw):
    """A nested estimator inside a meta-estimator's fixed arguments."""
    return {"$cls": path, "$kw": kw}


ALGOS: dict[int, dict] = {}


def add(entry, tasks, params=(), groups=("core",), gpu=False, scale="standard", max_rows=None, note=None):
    groups = (groups[0], "formats", *groups[1:])       # every algorithm can read every table format
    ALGOS[entry] = dict(entry=entry, tasks=tasks, params=list(params), groups=tuple(groups), gpu=gpu,
                        scale=scale, max_rows=max_rows, note=note)


R, K = "regression", "classification"
LM = "sklearn.linear_model"

# ---- Part 1: regression --------------------------------------------------------------
add(1, {R: sk(f"{LM}:LinearRegression")})
add(2, {R: sk(f"{LM}:LinearRegression", pre="poly")}, [I("degree", 2, 2, 5, [2, 3], to="pre")])
add(3, {R: sk(f"{LM}:Ridge")}, [ALPHA])
add(4, {R: sk(f"{LM}:Lasso")}, [ALPHA])
add(5, {R: sk(f"{LM}:ElasticNet")}, [ALPHA, F("l1_ratio", 0.5, 0, 1, [0.2, 0.5, 0.8])])
add(6, {R: sk(f"{LM}:BayesianRidge")})
add(7, {R: sk(f"{LM}:HuberRegressor", fixed={"max_iter": 500})}, [F("epsilon", 1.35, 1.0, 10)])
add(8, {R: sk(f"{LM}:RANSACRegressor")})
add(9, {R: sk(f"{LM}:TheilSenRegressor")}, max_rows=20000)
add(10, {R: sk(f"{LM}:QuantileRegressor", fixed={"solver": "highs"})},
    [F("quantile", 0.5, 0.01, 0.99), F("alpha", 0.01, 0, 100)], max_rows=20000)
add(11, {R: sk(f"{LM}:PoissonRegressor", fixed={"max_iter": 500}, target="nonnegative")}, [ALPHA])
add(12, {R: sk(f"{LM}:TweedieRegressor", fixed={"max_iter": 500}, target="nonnegative")},
    [F("power", 1.5, 0, 3), ALPHA])
add(13, {R: sk("sklearn.cross_decomposition:PLSRegression")}, [COMPONENTS])
add(14, {R: sk(f"{LM}:LinearRegression", pre="pca")}, [I("n_components", 2, 1, 200, to="pre")])
add(15, {R: sk("sklearn.svm:SVR")}, [SVM_C, KERNEL], max_rows=30000)
add(16, {R: sk("sklearn.neighbors:KNeighborsRegressor")}, [KNN])
add(17, {R: sk("sklearn.tree:DecisionTreeRegressor")}, [DEPTH], scale="none")
add(18, {R: sk("sklearn.gaussian_process:GaussianProcessRegressor", fixed={"normalize_y": True})}, max_rows=3000)
add(19, {R: sk("sklearn.kernel_ridge:KernelRidge")}, [ALPHA, KERNEL], max_rows=10000)
add(22, {R: sk(f"{LM}:Ridge", pre="spline")}, [I("n_knots", 5, 3, 30, to="pre"), ALPHA])
add(23, {R: sk(f"{LM}:Lars")})
add(24, {R: sk(f"{LM}:OrthogonalMatchingPursuit")})

# ---- Part 2: classification ----------------------------------------------------------
add(26, {K: sk(f"{LM}:LogisticRegression", fixed={"max_iter": 2000})}, [SVM_C])
add(27, {K: sk(f"{LM}:Perceptron")})
add(28, {K: sk("sklearn.neighbors:KNeighborsClassifier")}, [KNN])
add(29, {K: sk("sklearn.svm:SVC", fixed={"probability": True})}, [SVM_C, KERNEL], max_rows=30000)
add(30, {K: sk("sklearn.svm:LinearSVC", fixed={"max_iter": 5000})}, [SVM_C])
add(31, {K: sk("sklearn.naive_bayes:GaussianNB")})
add(32, {K: sk("sklearn.naive_bayes:MultinomialNB")}, [ALPHA], scale="minmax")
add(33, {K: sk("sklearn.naive_bayes:BernoulliNB")}, [ALPHA])
add(34, {K: sk("sklearn.naive_bayes:ComplementNB")}, [ALPHA], scale="minmax")
add(35, {K: sk("sklearn.tree:DecisionTreeClassifier")}, [DEPTH, C("criterion", "gini", ["gini", "entropy"])], scale="none")
add(37, {K: sk("sklearn.discriminant_analysis:LinearDiscriminantAnalysis")})
add(38, {K: sk("sklearn.discriminant_analysis:QuadraticDiscriminantAnalysis", fixed={"reg_param": 0.1})})
add(39, {K: sk("sklearn.neighbors:NearestCentroid")})
add(40, {K: sk(f"{LM}:RidgeClassifier")}, [ALPHA])
add(41, {K: sk(f"{LM}:SGDClassifier")}, [F("alpha", 1e-4, 1e-7, 1)])
add(42, {K: sk(f"{LM}:PassiveAggressiveClassifier")}, [SVM_C])
add(43, {K: sk("sklearn.gaussian_process:GaussianProcessClassifier")}, max_rows=2000)
_MLP = [I("hidden", 100, 4, 2000, label="hidden units"), F("alpha", 1e-4, 1e-7, 1), I("max_iter", 300, 5, 5000)]
add(44, {K: sk("sklearn.neural_network:MLPClassifier", hidden_arg=True)}, _MLP)
add(45, {K: sk("sklearn.multiclass:OneVsRestClassifier",
               fixed={"estimator": obj(f"{LM}:LogisticRegression", max_iter=2000)})})
add(49, {K: sk("sklearn.calibration:CalibratedClassifierCV",
               fixed={"estimator": obj("sklearn.svm:LinearSVC", max_iter=5000), "cv": 3})},
    [C("method", "sigmoid", ["sigmoid", "isotonic"])])

# ---- Part 3: ensembles ---------------------------------------------------------------
E = "sklearn.ensemble"
T = "sklearn.tree"
_TREE_ONLY = dict(scale="none")
add(50, {K: sk(f"{E}:BaggingClassifier"), R: sk(f"{E}:BaggingRegressor")},
    [I("n_estimators", 50, 2, 1000, [20, 50, 100])], **_TREE_ONLY)
_LAG = dict(adapter="lag")
add(51, {K: sk(f"{E}:RandomForestClassifier"), R: sk(f"{E}:RandomForestRegressor"),
         "forecasting": dict(_LAG, cls=f"{E}:RandomForestRegressor")}, [TREES, DEPTH], **_TREE_ONLY)
add(52, {K: sk(f"{E}:ExtraTreesClassifier"), R: sk(f"{E}:ExtraTreesRegressor"),
         "forecasting": dict(_LAG, cls=f"{E}:ExtraTreesRegressor")}, [TREES, DEPTH], **_TREE_ONLY)
add(53, {K: sk(f"{E}:AdaBoostClassifier"), R: sk(f"{E}:AdaBoostRegressor")}, [TREES, F("learning_rate", 1.0, 1e-3, 5)], **_TREE_ONLY)
add(54, {K: sk(f"{E}:GradientBoostingClassifier"), R: sk(f"{E}:GradientBoostingRegressor")},
    [I("n_estimators", 100, 10, 3000, [100, 200]), LR, I("max_depth", 3, 1, 30, [2, 3, 5])], **_TREE_ONLY)
add(55, {K: sk(f"{E}:HistGradientBoostingClassifier"), R: sk(f"{E}:HistGradientBoostingRegressor")},
    [I("max_iter", 100, 10, 3000, [100, 300]), LR, OI("max_depth", 0, 0, 100, label="max_depth (0 = unlimited)")], **_TREE_ONLY)
_BOOST = [TREES, LR, I("max_depth", 6, 1, 30, [3, 6, 9])]
add(56, {K: sk("xgboost:XGBClassifier", gbm="xgb"), R: sk("xgboost:XGBRegressor", gbm="xgb"),
         "forecasting": dict(_LAG, cls="xgboost:XGBRegressor", gbm="xgb")},
    _BOOST, groups=("core", "boost"), gpu=True, scale="none")
_LGB = [TREES, LR, I("num_leaves", 31, 2, 1024, [15, 31, 63])]
add(57, {K: sk("lightgbm:LGBMClassifier", gbm="lgb", fixed={"verbose": -1}),
         R: sk("lightgbm:LGBMRegressor", gbm="lgb", fixed={"verbose": -1}),
         "forecasting": dict(_LAG, cls="lightgbm:LGBMRegressor", gbm="lgb", fixed={"verbose": -1})},
    _LGB, groups=("core", "boost"), scale="none")
add(189, {"forecasting": dict(_LAG, cls="lightgbm:LGBMRegressor", gbm="lgb", fixed={"verbose": -1})},
    _LGB, groups=("core", "boost"), scale="none")
add(58, {K: sk("catboost:CatBoostClassifier", fixed={"verbose": 0, "allow_writing_files": False}),
         R: sk("catboost:CatBoostRegressor", fixed={"verbose": 0, "allow_writing_files": False})},
    [I("iterations", 300, 10, 5000, [200, 500]), LR, I("depth", 6, 1, 12, [4, 6, 8])],
    groups=("core", "catboost"), scale="none")
_VC = [("lr", obj(f"{LM}:LogisticRegression", max_iter=2000)), ("rf", obj(f"{E}:RandomForestClassifier", n_estimators=100)),
       ("nb", obj("sklearn.naive_bayes:GaussianNB"))]
_VR = [("ridge", obj(f"{LM}:Ridge")), ("rf", obj(f"{E}:RandomForestRegressor", n_estimators=100)),
       ("knn", obj("sklearn.neighbors:KNeighborsRegressor"))]
add(60, {K: sk(f"{E}:VotingClassifier", fixed={"estimators": _VC, "voting": "soft", "flatten_transform": False}),
         R: sk(f"{E}:VotingRegressor", fixed={"estimators": _VR})})
add(61, {K: sk(f"{E}:StackingClassifier", fixed={"estimators": _VC[:2], "cv": 3}),
         R: sk(f"{E}:StackingRegressor", fixed={"estimators": _VR[:2], "cv": 3})})
add(64, {K: sk(f"{E}:BaggingClassifier", fixed={"max_features": 0.5, "bootstrap": False}),
         R: sk(f"{E}:BaggingRegressor", fixed={"max_features": 0.5, "bootstrap": False})},
    [I("n_estimators", 50, 2, 1000)], **_TREE_ONLY)

# ---- Part 4: clustering --------------------------------------------------------------
Q = "clustering"
CL = "sklearn.cluster"
add(67, {Q: sk(f"{CL}:KMeans")}, [CLUSTERS])
add(69, {Q: sk(f"{CL}:MiniBatchKMeans")}, [CLUSTERS])
add(70, {Q: sk(f"{CL}:BisectingKMeans")}, [CLUSTERS])
add(73, {Q: sk(f"{CL}:AgglomerativeClustering", inductive=False)}, [CLUSTERS], max_rows=20000)
_EPS = [F("eps", 0.5, 1e-4, 1e4), I("min_samples", 5, 1, 1000)]
add(75, {Q: sk(f"{CL}:DBSCAN", inductive=False)}, _EPS)
add(76, {Q: sk(f"{CL}:HDBSCAN", inductive=False)}, [I("min_cluster_size", 5, 2, 10000)])
add(77, {Q: sk(f"{CL}:OPTICS", inductive=False)}, [I("min_samples", 5, 2, 1000)], max_rows=20000)
add(78, {Q: sk(f"{CL}:MeanShift")}, max_rows=10000)
_GMM = [I("n_components", 3, 1, 200)]
add(79, {Q: sk("sklearn.mixture:GaussianMixture")}, _GMM)
add(80, {Q: sk("sklearn.mixture:BayesianGaussianMixture", fixed={"max_iter": 500})}, _GMM)
add(82, {Q: sk(f"{CL}:SpectralClustering", inductive=False)}, [CLUSTERS], max_rows=5000)
add(83, {Q: sk(f"{CL}:AffinityPropagation")}, [F("damping", 0.7, 0.5, 0.99)], max_rows=5000)
add(84, {Q: sk(f"{CL}:Birch")}, [CLUSTERS])

# ---- Part 5: dimensionality reduction -----------------------------------------------
D = "dimreduction"
DC = "sklearn.decomposition"
MF = "sklearn.manifold"
add(89, {D: sk(f"{DC}:PCA")}, [COMPONENTS])
add(90, {D: sk(f"{DC}:IncrementalPCA")}, [COMPONENTS])
add(91, {D: sk(f"{DC}:KernelPCA")}, [COMPONENTS, C("kernel", "rbf", ["rbf", "linear", "poly", "sigmoid", "cosine"])], max_rows=5000)
add(92, {D: sk(f"{DC}:SparsePCA")}, [COMPONENTS], max_rows=20000)
add(93, {D: sk(f"{DC}:TruncatedSVD")}, [COMPONENTS])
add(95, {D: sk(f"{DC}:FastICA", fixed={"max_iter": 1000})}, [COMPONENTS])
add(96, {D: sk(f"{DC}:NMF", fixed={"max_iter": 1000})}, [COMPONENTS], scale="minmax")
add(97, {D: sk(f"{DC}:FactorAnalysis")}, [COMPONENTS])
add(100, {D: sk(f"{MF}:TSNE", inductive=False)}, [COMPONENTS, F("perplexity", 30, 2, 200)], max_rows=20000)
add(101, {D: sk("umap:UMAP")}, [COMPONENTS, I("n_neighbors", 15, 2, 200)], groups=("core", "umap"))
add(102, {D: sk(f"{MF}:Isomap")}, [COMPONENTS, KNN], max_rows=5000)
add(103, {D: sk(f"{MF}:LocallyLinearEmbedding")}, [COMPONENTS, I("n_neighbors", 10, 3, 200)], max_rows=5000)
add(104, {D: sk(f"{MF}:SpectralEmbedding", inductive=False)}, [COMPONENTS], max_rows=5000)
add(105, {D: sk(f"{MF}:MDS", inductive=False)}, [COMPONENTS], max_rows=2000)
add(107, {D: sk("sklearn.random_projection:GaussianRandomProjection")}, [COMPONENTS])

# ---- Part 7: anomaly detection ------------------------------------------------------
A = "anomaly"
add(129, {A: dict(adapter="simple", cls="zscore")})
add(130, {A: dict(adapter="simple", cls="iqr")})
add(132, {A: dict(adapter="simple", cls="mahalanobis")})
add(139, {A: dict(adapter="simple", cls="knn")}, [KNN], max_rows=50000)
add(142, {A: dict(adapter="simple", cls="pca_error")}, [I("n_components", 2, 1, 200)])
add(158, {A: dict(adapter="simple", cls="cusum")}, [F("k", 0.5, 0, 5)])
add(159, {A: dict(adapter="simple", cls="ewma")}, [F("lam", 0.2, 0.01, 1)])
add(133, {A: sk("sklearn.covariance:EllipticEnvelope", score="neg_samples")})
add(134, {A: sk(f"{E}:IsolationForest", score="neg_samples")}, [I("n_estimators", 200, 10, 3000)])
add(136, {A: sk("sklearn.neighbors:LocalOutlierFactor", score="neg_samples", fixed={"novelty": True})}, [KNN], max_rows=50000)
add(137, {A: sk("sklearn.svm:OneClassSVM", score="neg_samples")}, [F("nu", 0.05, 0.001, 0.99), KERNEL], max_rows=30000)
add(138, {A: sk("sklearn.svm:OneClassSVM", score="neg_samples", fixed={"kernel": "rbf"})}, [F("nu", 0.05, 0.001, 0.99)], max_rows=30000)
add(140, {A: sk("pyod.models.hbos:HBOS", score="decision")}, groups=("core", "pyod"))
add(141, {A: sk("pyod.models.ecod:ECOD", score="decision")}, groups=("core", "pyod"))

_TORCH = ("core", "torch")
_DEEP = [I("epochs", 30, 1, 2000, label="epochs"), F("lr", 1e-3, 1e-6, 1, label="learning rate"),
         I("hidden", 64, 4, 4096, label="hidden size")]
_SEQLEN = I("seq_len", 30, 2, 2000, label="sequence length")
_LAYERS = I("layers", 2, 1, 8, label="layers")
add(143, {A: dict(adapter="tabae", net="ae")}, _DEEP, groups=_TORCH, gpu=True)
add(110, {D: dict(adapter="tabae", net="ae", latent=True)}, _DEEP + [COMPONENTS], groups=_TORCH, gpu=True)
add(150, {A: dict(adapter="tabae", net="vae")}, _DEEP, groups=_TORCH, gpu=True)
add(151, {A: dict(adapter="tabae", net="svdd")}, _DEEP, groups=_TORCH, gpu=True)
add(144, {A: dict(adapter="seq", net="lstm_ae")}, _DEEP + [_SEQLEN], groups=_TORCH, gpu=True)
add(145, {A: dict(adapter="seq", net="lstm_pred")}, _DEEP + [_SEQLEN, _LAYERS], groups=_TORCH, gpu=True)

# ---- Part 8: forecasting ------------------------------------------------------------
Fc = "forecasting"
_ST = ("core", "stats")
_SP = I("seasonal_periods", 12, 2, 8760, label="season length")
add(165, {Fc: dict(adapter="stats", cls="sma")}, [I("window", 10, 1, 5000)], groups=_ST)
add(166, {Fc: dict(adapter="stats", cls="ses")}, groups=_ST)
add(167, {Fc: dict(adapter="stats", cls="holt")}, groups=_ST)
add(168, {Fc: dict(adapter="stats", cls="holt_winters")}, [_SP], groups=_ST)
add(169, {Fc: dict(adapter="stats", cls="ets")}, [_SP], groups=_ST)
add(170, {Fc: dict(adapter="stats", cls="ar")}, [I("lags", 5, 1, 500)], groups=_ST)
_PDQ = [I("p", 2, 0, 12), I("d", 1, 0, 3), I("q", 2, 0, 12)]
add(171, {Fc: dict(adapter="stats", cls="arima")}, _PDQ, groups=_ST)
add(172, {Fc: dict(adapter="stats", cls="sarima")}, _PDQ + [_SP], groups=_ST)

_S = "seq_classification"
_seq = lambda net, **kw: {Fc: dict(adapter="seq", net=net, **kw), _S: dict(adapter="seq", net=net, **kw)}
_RNN = _DEEP + [_SEQLEN, _LAYERS]
_CNN = _DEEP + [_SEQLEN]
add(190, {Fc: dict(adapter="seq", net="lstm")}, _RNN, groups=_TORCH, gpu=True)
add(191, _seq("tcn"), _CNN, groups=_TORCH, gpu=True)
add(199, {Fc: dict(adapter="seq", net="dlinear")}, _CNN, groups=_TORCH, gpu=True)
add(204, _seq("cnn1d"), _CNN, groups=_TORCH, gpu=True)
add(343, _seq("rnn"), _RNN, groups=_TORCH, gpu=True)
add(344, _seq("lstm"), _RNN, groups=_TORCH, gpu=True)
add(345, _seq("gru"), _RNN, groups=_TORCH, gpu=True)
add(346, _seq("bilstm"), _RNN, groups=_TORCH, gpu=True)
add(347, _seq("lstm", stacked=True), _DEEP + [_SEQLEN, I("layers", 3, 2, 8, label="layers")], groups=_TORCH, gpu=True)
add(357, _seq("wavenet"), _CNN, groups=_TORCH, gpu=True)

# ---- Parts 14 and 19: image classification ------------------------------------------
V = "image_classification"
_VIS = ("core", "torch", "vision")
_IMG = [I("epochs", 10, 1, 2000, label="epochs"), F("lr", 3e-4, 1e-6, 1, label="learning rate"),
        I("image_size", 224, 32, 1024, label="image size")]
_PRE = C("pretrained", "no", ["no", "yes"], label="ImageNet weights (download)")


def vision(entry, archs, default=None, fixed_size=None, pretrained=True):
    params = list(_IMG)
    if fixed_size:
        params = [p for p in params if p["name"] != "image_size"]
    if len(archs) > 1:
        params.insert(0, C("arch", default or archs[0], archs, label="variant"))
    if pretrained:
        params.append(_PRE)
    groups = _VIS + ("timm",) if archs[0].startswith("timm:") else _VIS
    add(entry, {V: dict(adapter="vision", arch=archs[0], fixed_size=fixed_size)}, params, groups=groups, gpu=True)


vision(317, ["lenet"], fixed_size=32, pretrained=False)
vision(318, ["alexnet"])
vision(320, ["vgg16", "vgg19"])
vision(321, ["googlenet"])
vision(322, ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
vision(323, ["wide_resnet50_2", "wide_resnet101_2"])
vision(324, ["resnext50_32x4d", "resnext101_32x8d"])
vision(325, ["densenet121", "densenet169", "densenet201"])
vision(327, ["squeezenet1_1", "squeezenet1_0"])
vision(328, ["mobilenet_v3_small", "mobilenet_v3_large", "mobilenet_v2"])
vision(329, ["shufflenet_v2_x1_0", "shufflenet_v2_x0_5"])
vision(330, ["efficientnet_b0", "efficientnet_b3", "efficientnet_v2_s"])
vision(331, ["mnasnet1_0", "mnasnet0_5"])
vision(333, ["regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf"])
vision(338, ["convnext_tiny", "convnext_small"])
vision(341, ["resnet18", "mobilenet_v3_small", "efficientnet_b0"], pretrained=False)  # always uses pretrained + frozen
ALGOS[341]["tasks"][V]["transfer"] = True
vision(481, ["vit_b_16", "vit_b_32"], fixed_size=224)
vision(483, ["swin_t", "swin_s"], fixed_size=224)


# ---- text --------------------------------------------------------------------------
TC, TG = "text_classification", "text_generation"
_TXT = ("core", "torch", "text")
_VEC = [I("max_features", 20000, 100, 2000000, label="vocabulary size"), I("ngram", 1, 1, 3, label="longest word group"), SVM_C]
add(386, {TC: dict(adapter="text_sk", vec="count")}, _VEC)
add(387, {TC: dict(adapter="text_sk", vec="tfidf")}, _VEC)
_HF = lambda default: [TXT("model", default, label="Hugging Face model"), I("epochs", 3, 1, 200, label="epochs"), F("lr", 2e-5, 1e-7, 1e-1, label="learning rate"),
                       I("max_length", 128, 8, 4096, label="max tokens per text")]
for _e, _m in ((369, "bert-base-uncased"), (370, "roberta-base"), (371, "distilbert-base-uncased"), (372, "answerdotai/ModernBERT-base"),
               (374, "facebook/bart-base"), (384, "sentence-transformers/all-MiniLM-L6-v2")):
    add(_e, {TC: dict(adapter="hf_text")}, _HF(_m), groups=_TXT, gpu=True, note="Downloads the pretrained weights on first use.")
_LLM = lambda default: [TXT("model", default, label="Hugging Face model"), I("epochs", 3, 1, 200, label="epochs"), F("lr", 5e-5, 1e-7, 1e-1, label="learning rate"),
                        I("block_size", 128, 8, 8192, label="max tokens per text")]
_LLMNOTE = "Downloads the base model on first use. Gated models such as Llama need a Hugging Face login first."
add(373, {TG: dict(adapter="llm")}, _LLM("distilgpt2"), groups=_TXT, gpu=True, note=_LLMNOTE)
add(375, {TG: dict(adapter="llm")}, _LLM("HuggingFaceTB/SmolLM2-135M-8k"), groups=_TXT, gpu=True, note=_LLMNOTE)
add(681, {TG: dict(adapter="llm")}, _LLM("HuggingFaceTB/SmolLM2-135M-8k"), groups=_TXT, gpu=True, note=_LLMNOTE + " Set a response column for instruction tuning.")
_LORA = lambda: _LLM("HuggingFaceTB/SmolLM2-135M-8k") + [I("rank", 8, 1, 256, label="LoRA rank")]
add(678, {TG: dict(adapter="llm", lora=True)}, _LORA(), groups=_TXT + ("peft",), gpu=True, note=_LLMNOTE)
add(680, {TG: dict(adapter="llm", lora=True, dora=True)}, _LORA(), groups=_TXT + ("peft",), gpu=True, note=_LLMNOTE)

# ---- more image backbones through timm -----------------------------------------------
vision(326, ["timm:xception41", "timm:xception65"])
vision(332, ["timm:ghostnet_100", "timm:ghostnetv2_100"])
vision(334, ["timm:hrnet_w18_small", "timm:hrnet_w18"])
vision(337, ["timm:repvgg_a0", "timm:repvgg_b0"])
vision(339, ["timm:nfnet_l0"])
vision(340, ["timm:fasternet_t0", "timm:efficientvit_b0", "timm:repvit_m1"])
vision(342, ["timm:resnet10t", "timm:mobilenetv4_conv_small"])
vision(482, ["timm:deit_tiny_patch16_224", "timm:deit3_small_patch16_224"], fixed_size=224)
vision(484, ["timm:twins_svt_small", "timm:coat_lite_tiny"], fixed_size=224)
vision(485, ["timm:maxvit_tiny_tf_224", "timm:coatnet_0_rw_224"], fixed_size=224)
vision(486, ["timm:mobilevit_xs", "timm:fastvit_t8"], fixed_size=256)
vision(487, ["timm:hiera_tiny_224"], fixed_size=224)
vision(492, ["timm:eva02_tiny_patch14_224"], fixed_size=224)

# ---- object detection ----------------------------------------------------------------
OD, SEG = "object_detection", "semantic_segmentation"
_DET = ("core", "torch", "vision")
_ULT = ("core", "torch", "vision", "ultra")
_AGPL = "Uses Ultralytics, licensed AGPL-3.0: check that licence before selling a product built on it."


def _detect(entry, archs, adapter, epochs, size, groups, note=None):
    params = [I("epochs", epochs, 1, 3000, label="epochs"), F("lr", 1e-3 if adapter == "detect_ultra" else 1e-4, 1e-6, 1, label="learning rate"),
              I("image_size", size, 96, 1600, label="image size"), C("pretrained", "no", ["no", "yes"], label="pretrained weights (download)")]
    if len(archs) > 1:
        params.insert(0, C("arch", archs[0], archs, label="variant"))
    add(entry, {OD: dict(adapter=adapter, arch=archs[0])}, params, groups=groups, gpu=True, note=note)


_detect(513, ["fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_resnet50_fpn_v2"], "detect_tv", 20, 512, _DET)
_detect(518, ["ssdlite320_mobilenet_v3_large", "ssd300_vgg16"], "detect_tv", 30, 320, _DET)
_detect(519, ["retinanet_resnet50_fpn_v2"], "detect_tv", 20, 512, _DET)
_detect(529, ["fcos_resnet50_fpn"], "detect_tv", 20, 512, _DET)
_detect(524, ["yolo11n", "yolo11s", "yolo11m", "yolov8n", "yolov8s", "yolov8m"], "detect_ultra", 50, 640, _ULT, _AGPL)
_detect(525, ["yolov10n", "yolov10s", "yolov9t", "yolov9s"], "detect_ultra", 50, 640, _ULT, _AGPL)
_detect(526, ["yolo26n", "yolo26s", "yolo26m"], "detect_ultra", 50, 640, _ULT, _AGPL)
_detect(534, ["rtdetr-l", "rtdetr-x"], "detect_ultra", 50, 640, _ULT, _AGPL)


# ---- segmentation --------------------------------------------------------------------
def _segment(entry, archs, groups):
    params = [C("arch", archs[0], archs, label="variant"), I("epochs", 30, 1, 3000, label="epochs"), F("lr", 3e-4, 1e-6, 1, label="learning rate"),
              I("image_size", 256, 64, 1600, label="image size")]
    if len(archs) == 1:
        params.pop(0)
    add(entry, {SEG: dict(adapter="segment", arch=archs[0])}, params, groups=groups, gpu=True)


_SMP = ("core", "torch", "vision", "smp")
_segment(542, ["tv:fcn_resnet50", "tv:fcn_resnet101"], _DET)
_segment(549, ["tv:deeplabv3_mobilenet_v3_large", "tv:deeplabv3_resnet50", "DeepLabV3Plus:resnet18", "DeepLabV3Plus:resnet34"], _SMP)
_segment(543, ["Unet:resnet18", "Unet:resnet34", "Unet:mobilenet_v2"], _SMP)
_segment(544, ["UnetPlusPlus:resnet18", "UnetPlusPlus:resnet34"], _SMP)
_segment(548, ["PSPNet:resnet18", "PSPNet:resnet34"], _SMP)
_segment(552, ["Segformer:mit_b0", "Segformer:mit_b1"], _SMP)

# ---- audio and video -----------------------------------------------------------------
AU, VI = "audio_classification", "video_classification"
_AUD = ("core", "torch", "audio")
_AP = [I("epochs", 30, 1, 3000, label="epochs"), F("lr", 1e-3, 1e-6, 1, label="learning rate"), I("sample_rate", 16000, 4000, 48000, label="sample rate"),
       F("seconds", 3.0, 0.2, 60, label="clip length (seconds)"), I("n_mels", 64, 16, 256, label="mel bands")]
add(658, {AU: dict(adapter="audio", feature="mfcc", net="cnn")}, _AP, groups=_AUD, gpu=True)
add(659, {AU: dict(adapter="audio", feature="mel", net="cnn")}, _AP, groups=_AUD, gpu=True)
add(667, {AU: dict(adapter="audio", feature="mel", net="tdnn")}, _AP, groups=_AUD, gpu=True)
add(670, {AU: dict(adapter="audio", feature="mfcc", net="cnn")}, _AP[:3] + [F("seconds", 1.0, 0.2, 10, label="clip length (seconds)"), _AP[4]], groups=_AUD, gpu=True)
_VID = ("core", "torch", "vision", "video")
_VP = [I("epochs", 20, 1, 1000, label="epochs"), F("lr", 3e-4, 1e-6, 1, label="learning rate"), I("frames", 16, 4, 128, label="frames per clip"), I("image_size", 112, 32, 512, label="frame size")]
add(601, {VI: dict(adapter="video", arch="c3d")}, _VP, groups=_VID, gpu=True)
add(603, {VI: dict(adapter="video", arch="r3d_18")}, [C("arch", "r3d_18", ["r3d_18", "r2plus1d_18", "mc3_18"], label="variant")] + _VP, groups=_VID, gpu=True)
add(349, {VI: dict(adapter="video", arch="cnn_lstm")}, _VP, groups=_VID, gpu=True)

# ---- generative and self-supervised ----------------------------------------------------
GE, RP = "image_generation", "representation"
_GP = [I("epochs", 50, 1, 5000, label="epochs"), F("lr", 2e-4, 1e-6, 1, label="learning rate"), C("image_size", "32", ["32", "64"], label="image size")]
_VISG = ("core", "torch", "vision")
for _e, _m in ((611, "vae"), (614, "gan"), (615, "dcgan"), (617, "wgan_gp"), (628, "ddpm"), (629, "ddim"), (635, "flow")):
    add(_e, {GE: dict(adapter="gen", method=_m)}, _GP, groups=_VISG, gpu=True)
add(612, {GE: dict(adapter="gen", method="beta_vae")}, _GP + [F("beta", 4.0, 0, 100, label="beta")], groups=_VISG, gpu=True)
_SP2 = [I("epochs", 30, 1, 3000, label="epochs"), F("lr", 1e-3, 1e-6, 1, label="learning rate"), I("image_size", 96, 32, 512, label="image size")]
for _e, _m in ((495, "rotation"), (496, "simclr"), (498, "byol"), (499, "barlow")):
    add(_e, {RP: dict(adapter="selfsup", method=_m)}, _SP2, groups=_VISG, gpu=True)

# ---- reinforcement learning --------------------------------------------------------------
RL = "reinforcement_learning"
_RLG = ("core", "torch", "rl")
for _e, _m, _p in ((687, "epsilon", [F("epsilon", 0.1, 0, 1)]), (688, "ucb", [F("c", 2.0, 0, 20)]), (689, "thompson", [])):
    add(_e, {RL: dict(adapter="bandit", method=_m)}, [I("steps", 5000, 10, 10000000, label="pulls")] + _p, groups=("core", "rl"))
_TAB = [I("episodes", 5000, 10, 5000000, label="episodes"), F("alpha", 0.1, 1e-4, 1, label="learning rate"), F("gamma", 0.99, 0, 1, label="discount"), I("max_steps", 500, 10, 100000, label="steps per episode")]
for _e, _m in ((692, "value_iteration"), (693, "mc"), (695, "q_learning"), (696, "sarsa"), (697, "expected_sarsa"), (698, "dyna_q")):
    add(_e, {RL: dict(adapter="rl_table", method=_m)}, _TAB, groups=("core", "rl"))
_SB = lambda lr: [I("timesteps", 50000, 100, 100000000, label="training steps"), F("lr", lr, 1e-6, 1, label="learning rate"), F("gamma", 0.99, 0, 1, label="discount")]
_S3, _C3 = "stable_baselines3", "sb3_contrib"
for _e, _c, _a, _lr, _kw in ((699, f"{_S3}:DQN", "discrete", 1e-3, dict(learning_starts=200, buffer_size=20000)), (702, f"{_C3}:QRDQN", "discrete", 1e-3, dict(learning_starts=200, buffer_size=20000)),
                             (704, f"{_S3}:A2C", "any", 7e-4, {}), (706, f"{_C3}:TRPO", "any", 1e-3, {}), (707, f"{_S3}:PPO", "any", 3e-4, {}),
                             (709, f"{_S3}:DDPG", "continuous", 1e-3, dict(learning_starts=200, buffer_size=20000)), (710, f"{_S3}:TD3", "continuous", 1e-3, dict(learning_starts=200, buffer_size=20000)),
                             (711, f"{_S3}:SAC", "continuous", 3e-4, dict(learning_starts=200, buffer_size=20000))):
    add(_e, {RL: dict(adapter="sb3", cls=_c, actions=_a, fixed=_kw)}, _SB(_lr), groups=_RLG, gpu=True)
add(708, {RL: dict(adapter="sb3", cls=f"{_C3}:RecurrentPPO", actions="any", policy="MlpLstmPolicy")}, _SB(3e-4), groups=_RLG, gpu=True)

# ---- graphs and recommenders -----------------------------------------------------------
GR, RC = "node_classification", "recommendation"
_GNN = [I("epochs", 200, 1, 20000, label="epochs"), F("lr", 0.01, 1e-5, 1, label="learning rate"), I("hidden", 64, 4, 4096, label="hidden size")]
for _e, _n in ((752, "cheb"), (753, "gcn"), (754, "sage"), (755, "gat"), (756, "gin"), (760, "transformer")):
    add(_e, {GR: dict(adapter="graph", net=_n)}, _GNN, groups=("core", "torch", "graph"), gpu=True)
_RCP = [I("epochs", 30, 1, 2000, label="epochs"), F("lr", 5e-3, 1e-6, 1, label="learning rate"), I("dim", 32, 2, 1024, label="embedding size")]
for _e, _n in ((767, "mf"), (769, "fm"), (771, "ncf"), (772, "two_tower")):
    add(_e, {RC: dict(adapter="recsys", net=_n)}, _RCP, groups=_TORCH, gpu=True)
add(766, {RC: dict(adapter="recsys", net="item_knn")}, [I("neighbours", 20, 1, 500, label="neighbours")], max_rows=200000)
add(768, {RC: dict(adapter="recsys", net="als")}, [I("rank", 20, 1, 500, label="factors"), F("reg", 0.1, 1e-6, 100, label="regularisation"), I("iterations", 15, 1, 200, label="iterations")], max_rows=500000)


# ---- status of everything else ------------------------------------------------------
_COMPONENT_PARTS = {9: "network building block", 10: "activation function", 11: "loss function",
                    12: "optimiser or schedule", 13: "regularisation or training trick"}
_TOOL_PARTS = {18: "image-processing operation", 33: "search or planning method",
               34: "evaluation or explainability method"}
_ADAPTER_PARTS = {16: "a language-model technique or an LLM variant with no training path here", 17: "a text-processing method with no training path here",
                  20: "a self-supervised method without its own trainer yet", 21: "a detector without a maintained training library that fits this app",
                  22: "a segmentation model that needs its own trainer or only runs pretrained", 23: "a tracking, pose or 3D method that mostly runs pretrained or needs video and sensor rigs",
                  24: "a video model without a maintained training library", 25: "a generative model needing far more compute or its own trainer",
                  26: "a multimodal model that runs pretrained rather than training on a dataset", 27: "a speech model needing its own pipeline",
                  28: "a compression technique applied to an existing model", 29: "an RL method without a maintained library here, or one needing a game or multi-agent setup",
                  30: "a robot method that needs a robot or physics simulator", 31: "a graph method without its own trainer yet", 32: "a recommender without its own trainer yet",
                  35: "a learning paradigm needing its own multi-dataset setup"}
BUILT_IN = {683: "This is built in: the Export step produces and verifies ONNX for supported models."}


def status_of(entry: int, part: int) -> tuple[str, str]:
    """Return (status, explanation). status: ready | component | tool | planned."""
    if entry in ALGOS:
        return "ready", "Trainable in this version."
    if entry in BUILT_IN:
        return "tool", BUILT_IN[entry]
    if part in _COMPONENT_PARTS:
        return "component", f"This is a {_COMPONENT_PARTS[part]}: it is used inside training, not trained on its own."
    if part in _TOOL_PARTS:
        return "tool", f"This is a {_TOOL_PARTS[part]}, not a model trained on a dataset."
    if part in _ADAPTER_PARTS:
        return "planned", f"Reference only: {_ADAPTER_PARTS[part]}."
    return "planned", "Reference only: this needs its own adapter, not built yet."


def get(entry: int) -> dict:
    return ALGOS[int(entry)]


def resolve_params(algo: dict, overrides: dict | None) -> dict:
    """Validated parameters: defaults, plus overrides that exist and pass their bounds."""
    overrides = overrides or {}
    out = {}
    for p in algo["params"]:
        v = overrides.get(p["name"], p["default"])
        if p["type"] in ("int", "optional_int"):
            v = int(v)
        elif p["type"] == "float":
            v = float(v)
        elif p["type"] == "choice" and v not in p["choices"]:
            raise ValueError(f"{p['name']}: {v!r} is not one of {p['choices']}")
        elif p["type"] == "text" and not re.fullmatch(r"[\w.-]+(/[\w.-]+)?", str(v)):
            raise ValueError(f"{p['name']}: {v!r} is not a valid model name")
        if p["type"] not in ("choice", "text") and not p["min"] <= v <= p["max"]:
            raise ValueError(f"{p['name']}: {v} is outside {p['min']}..{p['max']}")
        out[p["name"]] = v
    return out
