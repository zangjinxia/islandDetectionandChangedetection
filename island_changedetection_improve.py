'''
@author zangjinxia
@date 2020-12-3
@brief
该程序实现GF1/6/2和HY1C的遥感影像海岛的变化检测，输入参数为两景tif图和矢量变化结果文件
1、计算ndwi
2、对ndwi进行kmeans聚类（2类）图像分割，得到二值图，去除小面积
3、分别对两景影像进行处理，得到二值图
4、二值图相减。得到变化区域。
5、将变化区域的二值图转矢量面
6、去除背景矢量面，即矢量中面积最大的要素
'''
# coding=utf-8

import gdal
import os
from osgeo import ogr, osr
import numpy as np
from skimage import filters,morphology,measure
import matplotlib.pyplot as plt
import cv2


def read_img(filename):
    dataset = gdal.Open(filename)

    width = dataset.RasterXSize
    height = dataset.RasterYSize
    band = dataset.RasterCount
    im_data = dataset.ReadAsArray(0, 0, width, height)

    geotrans = dataset.GetGeoTransform()
    proj = dataset.GetProjection()
    # data = np.zeros([width, height, band])
    return im_data, proj, geotrans
def write_tiff(filename, proj, geotrans, data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64
    # 判断栅格数据的数据类型
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(data.shape) == 3:
        bands, height, width = data.shape
    else:
        bands = 1
        height, width = data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, width, height, bands, datatype)

    dataset.SetGeoTransform(geotrans)
    dataset.SetProjection(proj)

    if bands == 1:
        dataset.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(bands):
            dataset.GetRasterBand(i + 1).WriteArray(data[i])
    del dataset
def guiyihua(array,wid,hegt):
    ymax = 255
    ymin = 0
    xmax = max(map(max, array))
    xmin = min(map(min, array))
    for i in range(wid):
        for j in range(hegt):
            array[i][j] = round(((ymax - ymin) * (array[i][j] - xmin) / (xmax - xmin)) + ymin)
    return array

def NDWI(B1,B2):
    """
求归一化水体指数
    :param B1: 绿波段
    :param B2: 近红波段
    :return: NDWI矩阵
    """
    # result = (float(B1)-float(B2))/(float(B1)+float(B2))
    result1 = (B1 - B2) / (B1 + B2)
    return result1

def seg_kmeans_gray(img):
    # img = cv2.imread('ndwi.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    # 展平
    img_flat = img.reshape((img.shape[0] * img.shape[1], 1)) #降维，变为1列或者1行
    img_flat = np.float32(img_flat)

    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 聚类函数用法： 紧密度，结果标记，聚类中心组成的数组=kmeans（分类数据，分类数，预设的分类标签或者None，迭代停止的模式选择（有3种），重复试验算法次数，初始中心选择）
    compactness, labels, centers = cv2.kmeans(img_flat, 2, None, criteria, 10, flags)
    img_output = labels.reshape((img.shape[0], img.shape[1]))   #变为原来的维度

    # 显示结果
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('input')
    plt.subplot(122), plt.imshow(img_output, 'gray'), plt.title('kmeans')
    plt.show()
    return labels

def rasterToShp(rasterFile,shpFile):

    #注册驱动
    ogr.RegisterAll()
    strDriverName = "ESRI Shapefile"
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        print('驱动不可用')
        exit(1)
    #读取栅格影像
    # data,proj,geotrans = read_img(rasterFile)
    ds = gdal.Open(rasterFile, gdal.GA_ReadOnly)
    data = ds.GetRasterBand(1)
    #获取栅格图的空间参考，注意与ds.GetProjection的区别
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    # print(proj)
    space = proj.GetAttrValue('AUTHORITY', 1)
    # srs = ds.GetProjectionRef()
    # srs = osr.SpatialReference()
    # print(srs)
    maskband = data.GetMaskBand()
    # if data ==None:
    #     print('栅格图不存在')
    #     exit(1)
    #创建矢量文件
    oDS = oDriver.CreateDataSource(shpFile)  # 创建数据源
    if oDS == None:
        print('文件创建失败')
    # srs = None
    oLayer = oDS.CreateLayer('poly',proj, ogr.wkbPolygon)
    if oLayer == None:
        print('图层创建失败')
    oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # 创建一个叫FieldID的整型属性
    oLayer.CreateField(oFieldID, 0)
    oDefn = oLayer.GetLayerDefn()
    dst_field = 0
    prog_func = None
    options = []
    gdal.Polygonize(data,maskband,oLayer,dst_field,options,callback = prog_func)

def pol2line(polyfn,linefn):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    polyds = ogr.Open(polyfn,0)
    polyLayer = polyds.GetLayer()
    #创建输出文件
    if os.path.exists(linefn):
        driver.DeleteDataSource(linefn)
    lineds =driver.CreateDataSource(linefn)
    linelayer = lineds.CreateLayer(linefn,geom_type = ogr.wkbLineString)
    featuredefn = linelayer.GetLayerDefn()
    #获取ring到几何体
    #geomline = ogr.Geometry(ogr.wkbGeometryCollection)
    for feat in polyLayer:
        geom = feat.GetGeometryRef()
        ring = geom.GetGeometryRef(0)
        #geomcoll.AddGeometry(ring)
        outfeature = ogr.Feature(featuredefn)
        outfeature.SetGeometry(ring)
        linelayer.CreateFeature(outfeature)
        outfeature = None

def geopoint_advanced(shpfile):
    '''将所有点的坐标放到一个三维矩阵中'''
    driver = ogr.GetDriverByName('ESRI Shapefile')
    polyds = ogr.Open(shpfile, 0)
    lyr = polyds.GetLayer()
    featnumber = lyr.GetFeatureCount()
    for i in range(featnumber):
        feature = lyr.GetFeature(i)
        geo = feature.GetGeometryRef()
        ring = geo.GetGeometryRef(0)
        num_point = ring.GetPointCount()
        x = np.arange(num_point)
        y = np.arange(num_point)
        # arr = np.empty(featnumber,num_point,num_point)
        for j in range (num_point):
            x[j] = ring.GetX(j)
            y[j] = ring.GetY(j)
            arr1 = np.dstack((x,y))

        # appr_hand = measure.approximate_polygon(arr, tolerance=0.02)
        print(arr.shape)
        return arr
        print(arr)
        print(arr.shape)
def line_smooth(shpfile,resultshpfile):
    '''实现将有许多锯齿的线进行平滑，得到平滑后的线
    :param shpfile: 输入需平滑的线
    :param resultshpfile: 得到平滑后的结果SHP文件
    :param n: 平滑度
    '''
    driver = ogr.GetDriverByName('ESRI Shapefile')
    polyds = ogr.Open(shpfile, 0)
    lyr = polyds.GetLayer()
    featnumber = lyr.GetFeatureCount()
    srs = lyr.GetSpatialRef()
    #创建新矢量文件，存储平滑后的线
    ogr.RegisterAll()
    driver = ogr.GetDriverByName('ESRI Shapefile')

    if driver == None:
        print("%s 驱动不可用！\n", driver)
    oDS = driver.CreateDataSource(resultshpfile)  # 创建数据源
    if oDS == None:
        print("创建文件【%s】失败！", driver)
    # srspoly = osr.SpatialReference()
    oLayer = oDS.CreateLayer('line_smooth1',srs,ogr.wkbMultiLineString)
    if oLayer == None:
        print("图层创建失败！\n")
        #定义图层属性
    oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)  # 创建一个叫FieldID的整型属性
    oLayer.CreateField(oFieldID, 1)
    oDefn = oLayer.GetLayerDefn()  # 定义图层属性

   #遍历所有要素
    for i in range(featnumber):
        feature = lyr.GetFeature(i)
        geo = feature.GetGeometryRef()
        num_point = geo.GetPointCount()
        x = np.arange(num_point,dtype = float)
        y = np.arange(num_point,dtype = float)

        # 获取所遍历要素的所有坐标点
        for j in range (num_point):
            x[j] = geo.GetX(j)
            y[j] = geo.GetY(j)
            arr1 = np.dstack((x,y))
        #对该要素所有坐标点进行平滑
        arr = arr1.reshape(num_point,2)
        new_hand = arr.copy()
        for _ in range(5):
            new_hand = measure.subdivide_polygon(new_hand, degree=1)

        #得到平滑后的曲线
        line = ogr.Geometry(ogr.wkbLineString)
        num = new_hand.shape[0]
        for n in range(num):
            line.AddPoint(new_hand[n,0],new_hand[n,1])
        #存储线
        linesave = ogr.Feature(oDefn)
        geoline = ogr.CreateGeometryFromWkt(str(line))
        linesave.SetGeometry(geoline)
        oLayer.CreateFeature(linesave)

        #对矢量线进行展示
        # vp = VectorPlotter(True)
        # vp.plot(line, 'r-')
        # plt.show()
    oDS.Destroy()
def sealine_extract(rasterFile,linefile,size1,size2):
    """
实现自动提取海岸线功能
    :param rasterFile:需提取海岸线的遥感图像
    :param outPath: 输出结果文件的路径
    :param size1: 去除陆地上的小面积尺寸
    :param size2: 去除海洋中的小面积尺寸
    """
    outPath = os.path.dirname(linefile)
    data, proj, geotrans = read_img(rasterFile)
    b1 = np.array(data[0])
    b2 = np.array(data[1])
    b3 = np.array(data[2])
    b4 = np.array(data[3])
    print("开始计算ndwi")
    ndwi = NDWI(b2, b4)
    ndwi = np.array(ndwi)

    print("开始进行聚类")
    img = seg_kmeans_gray(ndwi)

    plt.imshow(img)
    plt.axis('off')  # 不显示坐标轴
    plt.show()
    print("阈值获取")
    thresh = filters.threshold_otsu(ndwi)  # 返回一个阈值
    watermask = (ndwi >= thresh) * 1.0  # 生成海水掩膜
    watermask = np.where(watermask==0,True,False)

    plt.imshow(watermask)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    print("去除小面积")
    #去除陆地上的小面积
    dst = morphology.remove_small_objects(watermask, min_size=size1, connectivity=1)  #conectivity的取值1表示4邻接，2表示8邻接
    #去除海水中的小面积
    dst = np.where(dst == 0,True,False)
    dst = morphology.remove_small_objects(dst, min_size=size2, connectivity=1)

    basename1 = os.path.basename(rasterFile)
    splittesxt1 = os.path.splitext(rasterFile)
    cfg1 = os.path.splitext(os.path.basename(rasterFile))[0]

    #二值图命名
    binaryName = cfg1+'BinaryImg.tif'
    # 转到存储数据的路径

    write_tiff(binaryName,proj,geotrans,dst)
    rasterFile = os.path.join(outPath,binaryName)

    return rasterFile

    # polyname = cfg1+'_'+'poly.shp'
    #
    # shpFile = os.path.join(outPath,polyname)
    #二值图转矢量面
    # rasterToShp(rasterFile,shpFile)
    #矢量面转线
    # lineFile = os.path.join(outPath,'island_line.shp')
    # pol2line(shpFile,linefile)
    # print("岸线提取完成！")
def area_rm(path):
    """
    实现矢量面积统计，将面积写道属性表中，并按照面积属性进行筛选，删除面积最大的要素。
    :param path:矢量文件
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    poly_DS = driver.Open(path,1)
    poly_lyr = poly_DS.GetLayer(0)
    prosrs = poly_lyr.GetSpatialRef()   #获取投影坐标信息
    #print("投影坐标系为:",prosrs)
    geosrs = prosrs.CloneGeogCS()    #获取地理坐标系
    #print("地理坐标系为:",geosrs)
    wkt = prosrs.ExportToWkt()     #将坐标系输出为字符串
    bool1 = wkt.find('UNIT["Metre"')
    bool2 = wkt.find('UNIT["metre"')
    bool3 = wkt.find('UNIT["Meter"')
    if bool1 == -1 and bool2 == -1 and bool3 == -1:
        print('请定义投影坐标！')
        # time.sleep(5)
        os._exit(-1)

    new_field = ogr.FieldDefn("Area", ogr.OFTString)
    # new_field.SetWidth(32)
    # new_field.SetPrecision(2)  # 设置面积精度
    poly_lyr.CreateField(new_field)
    featnumber = poly_lyr.GetFeatureCount()
    #将面积写道属性表中，并存到列表中
    list = []
    for i in range(featnumber):
        feature = poly_lyr.GetFeature(i)
        geom = feature.GetGeometryRef()
        area = geom.GetArea()
        area = round(area/1000000,2)
        list.append(area)
        feature.SetField("Area", area)  # 将面积添加到属性表中
        poly_lyr.SetFeature(feature)

    #在列表中找到最大的面积
    area_max = max(list)
    # 设置属性筛选条件
    strFilter = "Area = '" + str(area_max) + "'"
    #进行属性筛选
    poly_lyr.SetAttributeFilter(strFilter)
    #删除符合条件的要素
    pFeatureDef = poly_lyr.GetLayerDefn()
    # pLayerName = poly_lyr.GetName()
    pFieldName = "Area"
    # pFieldIndex = pFeatureDef.GetFieldIndex(pFieldName)
    for pFeature in poly_lyr:
        pFeatureFID = pFeature.GetFID()
        poly_lyr.DeleteFeature(int(pFeatureFID))
    strSQL = "REPACK " + str(poly_lyr.GetName())
    poly_DS.ExecuteSQL(strSQL, None, "")   #SQL语句
    pFeatureLayer = None
    poly_DS = None


if __name__ == '__main__':
    print('start')
    rasterPath_before = 'D:/AAdata/浙江2期/2016_caijain.tif'  # tif影像
    rasterPath_after = 'D:/AAdata/浙江2期/2020caijian.tif'
    change_result ='D:/AAdata/浙江2期/change.shp'
    size_land = 500
    size_sea = 500

    print("开始识别岛屿")
    rasterfile_before = sealine_extract(rasterPath_before,change_result,size_land,size_sea)
    rasterfile_after = sealine_extract(rasterPath_after, change_result, size_land, size_sea)

    #变化检测
    print('开始进行变化检测')
    databefore, projbefore, geotransbefore= read_img(rasterfile_before)
    dataafter, projafter, geotransafter = read_img(rasterfile_after)
    databefore = np.array(databefore)
    dataafter = np.array(dataafter)
    changedata =  dataafter-databefore
    changedata0 = np.where(changedata == -1, 0, changedata)
    changedata1 = np.where(changedata0 == 0,'nan',changedata0)
    # print(np.max(changedata1))
    # print(np.min(changedata1))
    #设置过程文件存放路径
    outPath = os.path.dirname(rasterPath_after)
    if (os.path.exists(outPath) == False):
        os.makedirs(outPath)
    os.chdir(outPath)

    write_tiff('change.tif',projafter,geotransafter,changedata1)
    print(rasterfile_before)
    print(rasterfile_after)
    rasterToShp('change.tif', change_result)

    #删除大面积无效矢量区域
    print('删除大面积无效矢量区域')
    area_rm(change_result)
    print('变化检测完成')




