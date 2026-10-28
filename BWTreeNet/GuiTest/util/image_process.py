import cv2
import numpy as np
def resize_img(img_data, resize_img_hei, resize_img_wid, first_dim_channel = True):
    if len(img_data.shape) ==3:
        if first_dim_channel:
            band_num = img_data.shape[0]
            new_img_data = np.zeros([band_num, resize_img_hei, resize_img_wid])
            for b in range(band_num):
                new_band = img_data[b]
                new_band = cv2.resize(
                    new_band, (resize_img_hei, resize_img_wid))
                new_img_data[b] = new_band
        else:
            band_num = img_data.shape[-1]
            new_img_data = np.zeros([resize_img_hei, resize_img_wid, band_num])
            for b in range(band_num):
                new_band = img_data[:, :, b]
                new_band = cv2.resize(
                    new_band, (resize_img_hei, resize_img_wid))
                new_img_data[:, :, b] = new_band
    elif len(img_data.shape) == 2:
        new_img_data = cv2.resize(img_data, (resize_img_hei, resize_img_wid))
    else:
        print('The shape of image is ',str(len(img_data.shape)),' dims, please check')
        return False
    return new_img_data


def write_img(img_data, img_path):
    tif_img_type = ['tif', 'tiff']
    other_img_type = ['jpg', 'png', 'bmp']
    if len(img_data.shape) > 2:
        if img_data.shape[2] > img_data.shape[0]:
            img_wid = img_data.shape[1]
            img_hei = img_data.shape[2]
            band_num = img_data.shape[0]
        else:
            img_wid = img_data.shape[0]
            img_hei = img_data.shape[1]
            band_num = img_data.shape[2]

        img_suffix = img_path.split('.')[-1]
        if img_suffix.lower() in other_img_type:
            if img_data.shape[2] > img_data.shape[0]:
                img_data = img_data.transpose(2, 0, 1)
            if band_num > 3:
                print('image is over 3 bands')
                return
            if img_data.max() > 255:
                print('the value of data is bigger than 255')
            import cv2
            cv2.imwrite(img_path, img_data)
        if img_suffix.lower() in tif_img_type:
            from osgeo import gdal
            if img_data.shape[2] < img_data.shape[0]:
                img_data = img_data.transpose(2, 0, 1)
            gdal.SetConfigOption("GDAL_FILENAME_IF_UTF8", "NO")
            gdal.SetConfigOption("SHAPE_ENCODING", "")
            if img_data.max() < 256:
                tiff_datatype = gdal.GDT_Byte
            else:
                tiff_datatype = gdal.GDT_UInt16
            driver = gdal.GetDriverByName("GTiff")
            dataset = driver.Create(
                img_path, img_hei, img_wid, band_num, tiff_datatype)
            for b in range(band_num):
                new_band = img_data[b]
                dataset.GetRasterBand(b+1).WriteArray(new_band)
            del dataset
    else:
        img_wid = img_data.shape[0]
        img_hei = img_data.shape[1]
        band_num = 1
        img_suffix = img_path.split('.')[-1]
        if img_suffix.lower() in other_img_type:
            import numpy as np
            img_data = img_data[:, :, np.newaxis]
            if img_data.max() > 255:
                print('the value of data is bigger than 255')
            import cv2
            cv2.imwrite(img_path, img_data)
        if img_suffix.lower() in tif_img_type:
            from osgeo import gdal
            gdal.SetConfigOption("GDAL_FILENAME_IF_UTF8", "NO")
            gdal.SetConfigOption("SHAPE_ENCODING", "")
            if img_data.max() < 256:
                tiff_datatype = gdal.GDT_Byte
            else:
                tiff_datatype = gdal.GDT_UInt16
            driver = gdal.GetDriverByName("GTiff")
            dataset = driver.Create(
                img_path, img_hei, img_wid, band_num, tiff_datatype)
            import numpy as np
            img_data = img_data[np.newaxis, :, :]
            for b in range(band_num):
                new_band = img_data[b]
                dataset.GetRasterBand(b+1).WriteArray(new_band)
            del dataset


# import numpy as np
# img_path = 'fdsafdsafdsaf/.tiff'
# img_data = np.zeros([128, 256, 3])
# write_img(img_data,img_path)
