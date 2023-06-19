"""
Object pop-up models with and without class prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .pointnetv2 import PointNetv2, square_distance
from .pointnetv2_segmentation import PointNetv2Seg
from .decoders import DecoderPointNet2, DecoderPointNet2Seg


class ObjectPopup(nn.Module):
    def __init__(
        self,
        encoder_subject_params,
        encoder_object_params,
        encoder_subjectobject_params,
        decoder_params, canonical_obj_keypoints,
        decoder_type="offsets", n_points=None, sbj_local_n_points=500,
        with_classifier=False, with_sbjobj_enc=True
    ):
        super().__init__()
        # ==> ENCODER SUBJECT
        self.encoder_subject = PointNetv2(**encoder_subject_params)
        self.sbj_local_n_points = sbj_local_n_points
        enc_sbj_out_dim = self.encoder_subject.out_dim
        # <===

        # ==> CENTER DECODER
        self.decoder_center = DecoderPointNet2(
            in_dim=self.encoder_subject.out_dim,
            out_dim=3,
            hidden_dim=256
        )
        # <===

        # ==> CLASSIFIER OBJECT
        self.with_classifier = with_classifier
        if with_classifier:
            self.classifier = DecoderPointNet2(
                in_dim=enc_sbj_out_dim,
                out_dim=encoder_object_params["n_onehot_classes"],
                hidden_dim=encoder_object_params["classifier_hidden_layer"]
            )
        # <===

        # ==> ENC SBJ-OBJ
        if with_sbjobj_enc:
            self.encoder_sbjobj = PointNetv2Seg(**encoder_subjectobject_params)
            enc_sbjobj_out_dim = self.encoder_sbjobj.out_dim
        # <===

        # ==> ENCODER OBJECT
        self.encoder_object_type = "onehot"
        self.n_onehot_classes = encoder_object_params["n_onehot_classes"]
        self.encoder_object = nn.Sequential(
            nn.Linear(self.n_onehot_classes, 128),
            nn.ReLU(),
        )
        self.encoder_object_out_dim = 128
        # <===

        # ==> DECODERS
        self.decoder_type = decoder_type
        self.n_points = 1500 if n_points is None else n_points
        self.decoder_feature_size = enc_sbj_out_dim + self.encoder_object_out_dim
        self.with_sbjobj_enc = with_sbjobj_enc
        if self.with_sbjobj_enc:
            self.decoder_feature_size += enc_sbjobj_out_dim

        if self.decoder_type == "offsets":
            self.decoder = DecoderPointNet2Seg(in_dim=self.decoder_feature_size + 9, **decoder_params)
        elif self.decoder_type == "Rt":
            self.decoder_R = DecoderPointNet2Seg(in_dim=self.decoder_feature_size + 9 * self.n_points, out_dim=9, **decoder_params)
            self.decoder_t = DecoderPointNet2Seg(in_dim=self.decoder_feature_size + 9 * self.n_points, out_dim=3, **decoder_params)

        # load keypoints in canonical pose
        self.canonical_obj_keypoints = torch.zeros((self.n_onehot_classes, self.n_points, 3), dtype=torch.float,
                                                   requires_grad=False)
        for class_id in range(self.n_onehot_classes):
            self.canonical_obj_keypoints[class_id] = \
                torch.tensor(canonical_obj_keypoints[class_id]["cartesian"], dtype=torch.float, requires_grad=False)

    def encode_1st_stage(
        self, subject,
    ):
        batch_size, sbj_n_points, _ = subject.shape
        self.canonical_obj_keypoints = self.canonical_obj_keypoints.to(subject.device)

        # encode subject
        enc_sbj_global = self.encoder_subject(subject)

        # pred center
        pred_center = self.decoder_center(enc_sbj_global).view(batch_size, 1, 3)  # B x 1 x 3

        # encode object
        pred_class = None
        if self.with_classifier:
            pred_class = self.classifier(enc_sbj_global)

        return enc_sbj_global, pred_center, pred_class

    def encode_2nd_stage(
        self, subject, pred_center, pred_class, obj_classids,
        obj_center=None, obj_scales=None, obj_keypoints=None
    ):
        batch_size, sbj_n_points, _ = subject.shape

        # encode class
        if self.with_classifier:
            if obj_classids is None:
                classids = pred_class.argmax(dim=1)
            else:
                # convert class ids to one hot vector
                classids = obj_classids.to(subject.device)
        else:
            # convert class ids to one hot vector
            classids = obj_classids.to(subject.device)
        # convert class ids to one hot vector
        classes_enc = F.one_hot(classids, num_classes=self.n_onehot_classes).float()
        enc_object = self.encoder_object(classes_enc)

        # find local center
        if obj_center is None:
            local_center = pred_center
        else:
            local_center = obj_center.view(batch_size, 1, 3).to(subject.device)

        # encode subject + object
        # local sbj neigborhood
        sbj_local_dist = square_distance(local_center, subject)  # B x 1 x N_points
        sbj_local_idx = sbj_local_dist.squeeze(1).sort(dim=1)[1][:, :self.sbj_local_n_points]  # B x N_local_points
        sbj_local = subject[
            torch.arange(0, batch_size, device=subject.device).view(batch_size, 1),
            sbj_local_idx
        ]
        # local object neighborhood
        # load keypoints from internal mapping
        if obj_keypoints is None:
            classes_one_hot = F.one_hot(classids, num_classes=self.n_onehot_classes).view(batch_size, -1).float()
            obj_keypoints = torch.matmul(
                classes_one_hot,
                self.canonical_obj_keypoints.permute(2, 0, 1)
            )
            obj_keypoints = obj_keypoints.permute(1, 2, 0)

            # rescale keypoints to align with PC
            if obj_scales is not None:
                obj_keypoints = obj_keypoints * obj_scales.reshape(batch_size, 1, 1).float()

            obj_keypoints = obj_keypoints - obj_keypoints.mean(dim=1, keepdims=True) + local_center
        else:
            obj_keypoints = obj_keypoints.to(subject.device)

        # concatenate sbj and obj
        if self.with_sbjobj_enc:
            mask_sbj = torch.zeros((batch_size, sbj_local.size(1), 1), dtype=torch.float, device=subject.device)
            mask_obj = torch.ones((batch_size, obj_keypoints.size(1), 1), dtype=torch.float, device=subject.device)
            mask_sbjobj = torch.cat([mask_sbj, mask_obj], dim=1)
            pc_sbjobj = torch.cat([sbj_local, obj_keypoints], dim=1)
            in_sbjobj = torch.cat([pc_sbjobj, mask_sbjobj], dim=2)
            # encode
            enc_sbjobj = self.encoder_sbjobj(in_sbjobj)
            # keep only obj points
            enc_sbjobj = enc_sbjobj[:, sbj_local.size(1):]
        else:
            enc_sbjobj = None

        return enc_object, enc_sbjobj, obj_keypoints

    def decode(
        self, enc_subject, enc_object, enc_sbjobj, pred_center, obj_keypoints,
    ):
        batch_size = enc_subject.size(0)
        n_points = obj_keypoints.size(1)

        if self.decoder_type == "offsets":
            object_keypoints_t = torch.cat(
                [obj_keypoints, torch.sin(2 * np.pi * obj_keypoints), torch.cos(2 * np.pi * obj_keypoints)],
            dim=2).float()  # B_SIZE x N_POINTS x 9

            # repeat features for each point
            enc_object = enc_object.repeat_interleave(n_points, dim=0)  # B_SIZE * N_POINTS x F2
            enc_object = enc_object.reshape(batch_size, n_points, -1)

            # B_SIZE x N_POINTS x F1+F2+F3+9
            # repeat features for each point
            enc_subject = enc_subject.repeat_interleave(n_points, dim=0)  # B_SIZE * N_POINTS x F1
            enc_subject = enc_subject.reshape(batch_size, n_points, -1)
            features = torch.cat([enc_subject, enc_object], dim=2)

            if self.with_sbjobj_enc:
                features = torch.cat([features, enc_sbjobj], dim=2)
            else:
                pass

            features = torch.cat([features, object_keypoints_t], dim=2)
            features = features.reshape(batch_size * n_points, -1)
            offsets = self.decoder(features)
            offsets = offsets.reshape(batch_size, n_points, -1)

            return {"offsets": offsets, "obj_center": pred_center}
        elif self.decoder_type == "Rt":
            # Aggregate features for points using max (B x n_points x F) -> (B x F)
            enc_sbjobj = torch.max(enc_sbjobj, dim=1)[0]

            # Create point coord features
            object_keypoints_t = torch.cat(
                [obj_keypoints, torch.sin(2 * np.pi * obj_keypoints), torch.cos(2 * np.pi * obj_keypoints)],
            dim=2).float()  # B_SIZE x N_POINTS x 9
            object_keypoints_t = object_keypoints_t.reshape(batch_size, n_points * 9)

            # Concatenate features
            features = torch.cat([enc_subject, enc_object, enc_sbjobj, object_keypoints_t], dim=1)

            # Decode R,t
            R = self.decoder_R(features)
            t = self.decoder_t(features)

            return {"R": R, "t": t, "obj_center": pred_center}

    def encode(
        self, subject, obj_classids, obj_center=None, obj_scales=None, obj_keypoints=None
    ):
        enc_subject, pred_center, pred_class = self.encode_1st_stage(
            subject
        )

        enc_object, enc_sbjobj, obj_keypoints = self.encode_2nd_stage(
            subject, pred_center, pred_class, obj_classids, obj_center=obj_center,
            obj_scales=obj_scales, obj_keypoints=obj_keypoints
        )

        return enc_subject, enc_object, enc_sbjobj, pred_center, pred_class, obj_keypoints

    def forward(
        self, subject, obj_classids, obj_keypoints=None,
        obj_scales=None, obj_center=None
    ):
        enc_subject, enc_object, enc_sbjobj, pred_center, pred_class, obj_keypoints = self.encode(
            subject, obj_classids, obj_keypoints=obj_keypoints,
            obj_center=obj_center, obj_scales=obj_scales,
        )

        predictions = self.decode(
            enc_subject, enc_object, enc_sbjobj, pred_center, obj_keypoints
        )

        if self.with_classifier:
            predictions["obj_class"] = pred_class

        return predictions
