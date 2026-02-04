
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.root_dir = 'F:/lilaoshi/Dataset/CD/LEVIR-CD-256/'  # 数据输入路径
        elif data_name == 'WHU':
            self.root_dir = 'F:/lilaoshi/Dataset/CD/WHU-CD-256/'
        elif data_name == 'SYSU':
            self.root_dir = 'F:/lilaoshi/Dataset/CD/SYSU-CD/'
        elif data_name == 'CDD':
            self.root_dir = 'F:/lilaoshi/Dataset/CD/CDD-CD-256/'
        elif data_name == 'DSIFN':
            self.root_dir = 'F:/lilaoshi/Dataset/CD/DSIFN-CD-256/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

