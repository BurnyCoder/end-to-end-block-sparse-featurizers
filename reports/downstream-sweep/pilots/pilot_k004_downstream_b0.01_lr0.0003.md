# Pilot: pilot_k004_downstream_b0.01_lr0.0003

Development-only seed 900; not a main-study or test result. See the [full inventory and limitations](../pilots.md).

```json
{
  "name": "pilot_k004_downstream_b0.01_lr0.0003",
  "cell": "k004_downstream_b0.01",
  "label": "KL + 0.01 downstream MSE",
  "k": 4,
  "lr": 0.0003,
  "seed": 900,
  "state": {
    "cell": {
      "k": 4,
      "objective": "downstream",
      "beta": 0.01
    },
    "seed": 900,
    "epoch": 1,
    "updates": 16,
    "images_seen": 510,
    "microbatch": 8,
    "status": "complete",
    "elapsed_seconds": 62.5,
    "train_seconds": 59.09399999998277,
    "dev_seconds": 2.5160000000032596,
    "peak_vram_bytes": 1199974912,
    "source_revision": "58751c0d3d0fa85727960cf115facfe18e3a2e12",
    "initial_sha256": "e358b0b361ca7aca22164b40b8ba77896e3a3291c54a5ebe5c6de5a2f8c24d5e"
  },
  "development": {
    "kl": 2.443610429763794,
    "ce_delta": 2.4516239166259766,
    "accuracy": 0.2280000001192093,
    "teacher_accuracy": 0.9919999837875366,
    "agreement": 0.2280000001192093,
    "images": 250,
    "layer_mse": [
      [
        1.6228631734848022,
        1.298318862915039,
        14.345016479492188
      ],
      [
        5.599618911743164,
        5.360445499420166,
        14.97521686553955
      ],
      [
        11.64286994934082,
        11.454011917114258,
        19.04592514038086
      ],
      [
        29.964998245239258,
        30.111186981201172,
        24.234615325927734
      ],
      [
        77.90449523925781,
        78.63465118408203,
        49.2810173034668
      ],
      [
        232.2655792236328,
        234.30650329589844,
        152.2562713623047
      ],
      [
        1426.0433349609375,
        1435.136962890625,
        1069.5816650390625
      ]
    ],
    "layer_nmse": [
      [
        0.9009835124015808,
        0.9126113057136536,
        0.8620175719261169
      ],
      [
        1.6273741722106934,
        1.7331018447875977,
        0.8768103718757629
      ],
      [
        1.4902424812316895,
        1.5195879936218262,
        1.0240436792373657
      ],
      [
        1.5111879110336304,
        1.521631121635437,
        1.1325820684432983
      ],
      [
        1.6031478643417358,
        1.6042096614837646,
        1.5394279956817627
      ],
      [
        1.5213438272476196,
        1.5147161483764648,
        2.0668911933898926
      ],
      [
        2.1210274696350098,
        2.110182285308838,
        2.906705379486084
      ]
    ],
    "insertion_nmse": 0.9009836316108704,
    "downstream_mse": 297.23681640625,
    "epoch": 1,
    "images_seen": 510
  },
  "history": [
    {
      "update": 1,
      "epoch": 0,
      "images_seen": 32,
      "loss": 6.57652223110199,
      "kl": 3.3756935000419617,
      "insertion_nmse": 0.9228818565607071,
      "downstream_mse": 320.08287048339844,
      "lr": 0.0002974060128544361,
      "gradient_norm": 5.091587066650391,
      "elapsed_seconds": 3.672000000020489
    },
    {
      "update": 2,
      "epoch": 0,
      "images_seen": 64,
      "loss": 6.6437908411026,
      "kl": 3.4705689549446106,
      "insertion_nmse": 0.832100436091423,
      "downstream_mse": 317.3222122192383,
      "lr": 0.0002897237368890237,
      "gradient_norm": 5.348817825317383,
      "elapsed_seconds": 7.312999999994645
    },
    {
      "update": 3,
      "epoch": 0,
      "images_seen": 96,
      "loss": 6.578853011131287,
      "kl": 3.4309805035591125,
      "insertion_nmse": 0.9303363710641861,
      "downstream_mse": 314.7872543334961,
      "lr": 0.0002772483976608436,
      "gradient_norm": 5.21094274520874,
      "elapsed_seconds": 10.844000000011874
    },
    {
      "update": 4,
      "epoch": 0,
      "images_seen": 128,
      "loss": 6.304536938667297,
      "kl": 3.15490984916687,
      "insertion_nmse": 0.8539725095033646,
      "downstream_mse": 314.96273040771484,
      "lr": 0.0002604594154601839,
      "gradient_norm": 5.345860481262207,
      "elapsed_seconds": 14.469000000011874
    },
    {
      "update": 5,
      "epoch": 0,
      "images_seen": 160,
      "loss": 7.111894607543945,
      "kl": 3.9522162675857544,
      "insertion_nmse": 0.9233079850673676,
      "downstream_mse": 315.96783447265625,
      "lr": 0.0002400019814576463,
      "gradient_norm": 5.684668064117432,
      "elapsed_seconds": 18.14100000000326
    },
    {
      "update": 6,
      "epoch": 0,
      "images_seen": 192,
      "loss": 6.375460863113403,
      "kl": 3.299042820930481,
      "insertion_nmse": 0.8328093141317368,
      "downstream_mse": 307.6418151855469,
      "lr": 0.00021666226336928708,
      "gradient_norm": 5.202649116516113,
      "elapsed_seconds": 21.76600000000326
    },
    {
      "update": 7,
      "epoch": 0,
      "images_seen": 224,
      "loss": 5.887426137924194,
      "kl": 2.789599895477295,
      "insertion_nmse": 0.9303857833147049,
      "downstream_mse": 309.78263092041016,
      "lr": 0.00019133719347217733,
      "gradient_norm": 5.196289539337158,
      "elapsed_seconds": 25.485000000015134
    },
    {
      "update": 8,
      "epoch": 0,
      "images_seen": 256,
      "loss": 5.778480648994446,
      "kl": 2.7881941199302673,
      "insertion_nmse": 0.8727413415908813,
      "downstream_mse": 299.0286636352539,
      "lr": 0.000165,
      "gradient_norm": 5.816587448120117,
      "elapsed_seconds": 29.28200000000652
    },
    {
      "update": 9,
      "epoch": 0,
      "images_seen": 288,
      "loss": 6.517574787139893,
      "kl": 3.5024563670158386,
      "insertion_nmse": 0.7743332535028458,
      "downstream_mse": 301.5118408203125,
      "lr": 0.00013866280652782267,
      "gradient_norm": 6.100938320159912,
      "elapsed_seconds": 32.953000000008615
    },
    {
      "update": 10,
      "epoch": 0,
      "images_seen": 320,
      "loss": 5.7043644189834595,
      "kl": 2.6503461599349976,
      "insertion_nmse": 0.8440720438957214,
      "downstream_mse": 305.4018249511719,
      "lr": 0.00011333773663071288,
      "gradient_norm": 6.150400161743164,
      "elapsed_seconds": 36.578000000008615
    },
    {
      "update": 11,
      "epoch": 0,
      "images_seen": 352,
      "loss": 5.577837944030762,
      "kl": 2.586371958255768,
      "insertion_nmse": 0.8606666475534439,
      "downstream_mse": 299.14659881591797,
      "lr": 8.999801854235373e-05,
      "gradient_norm": 5.152712345123291,
      "elapsed_seconds": 40.328000000008615
    },
    {
      "update": 12,
      "epoch": 0,
      "images_seen": 384,
      "loss": 5.635889649391174,
      "kl": 2.5849441289901733,
      "insertion_nmse": 0.8740488737821579,
      "downstream_mse": 305.0945510864258,
      "lr": 6.954058453981609e-05,
      "gradient_norm": 5.8758544921875,
      "elapsed_seconds": 44.14100000000326
    },
    {
      "update": 13,
      "epoch": 0,
      "images_seen": 416,
      "loss": 5.533483505249023,
      "kl": 2.4641023874282837,
      "insertion_nmse": 0.8106079995632172,
      "downstream_mse": 306.9381103515625,
      "lr": 5.275160233915637e-05,
      "gradient_norm": 5.403927803039551,
      "elapsed_seconds": 47.969000000011874
    },
    {
      "update": 14,
      "epoch": 0,
      "images_seen": 448,
      "loss": 5.641523122787476,
      "kl": 2.6603862047195435,
      "insertion_nmse": 0.8733308762311935,
      "downstream_mse": 298.1137008666992,
      "lr": 4.027626311097629e-05,
      "gradient_norm": 5.66684103012085,
      "elapsed_seconds": 51.78200000000652
    },
    {
      "update": 15,
      "epoch": 0,
      "images_seen": 480,
      "loss": 5.598296523094177,
      "kl": 2.658634662628174,
      "insertion_nmse": 0.8987209498882294,
      "downstream_mse": 293.96617889404297,
      "lr": 3.259398714556389e-05,
      "gradient_norm": 5.9314398765563965,
      "elapsed_seconds": 55.562999999994645
    },
    {
      "update": 16,
      "epoch": 0,
      "images_seen": 510,
      "loss": 6.01396328608195,
      "kl": 2.979238224029541,
      "insertion_nmse": 0.9716231425603231,
      "downstream_mse": 303.47252197265624,
      "lr": 2.9999999999999997e-05,
      "gradient_norm": 6.961751461029053,
      "elapsed_seconds": 59.203000000008615
    }
  ]
}
```
