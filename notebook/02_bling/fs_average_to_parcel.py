

mapper_dir = "/mnt/raid/bling/share/mappers"

mapper_file = os.path.join(mapper_dir, f"{subject_id}_fsaverage_mapper.hdf")

mapper = load_hdf5_sparse_array(mapper_file, key="voxel_to_fsaverage")

projected_en = timescale_en @ mapper.T
projected_zh = timescale_zh @ mapper.T

projected_pred_acc_en = pred_acc_en @ mapper.T
projected_pred_acc_zh = pred_acc_zh @ mapper.T