class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'spacenet':
            return '/home/mj/data/work_road/data/SpaceNet/spacenet/result_3m/'
        elif dataset == 'DeepGlobe':
            return '/home/mj/data/work_road/data/DeepGlobe/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
