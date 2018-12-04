import tensorflow as tf
import matplotlib.pyplot as plt
import sys

# 输入提示
location_to_be_input = input("请输入测试图片的文件夹位置：")
number_to_be_input = input("请输入测试图片的数量：")

# 图片处理次数
cycle = 1;

#读取训练好的神经网络
# 读取label
label_lines = [line.rstrip() for line
               in tf.gfile.GFile("./labels.txt")]

# 读取graph
with tf.gfile.FastGFile("./graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

# 记录一张图片的三个工件合格与否
output_data_arr = [0,0,0]

# 循环体
while cycle < (int(number_to_be_input) + 1):

  # 根据输入的路径定位图片
  image_path = location_to_be_input + "\\%s.jpg" % (str(cycle))

  #读取图片
  image_value = tf.read_file(image_path)
  # 图片解码，彩色则channels为3，黑白则channels为1
  img = tf.image.decode_jpeg(image_value, channels=3)

  # 分割用到的常量
  x = 545
  y = 1085
  z = 1640

  # 定义分割次数
  count = 1

  while count < 4:

    # 图像分割
    if count == 1:
      cropped_image = tf.image.crop_to_bounding_box(img, 310, x, 1560, 310)
    elif count == 2:
      cropped_image = tf.image.crop_to_bounding_box(img, 310, y, 1560, 310)
    elif count == 3:
      cropped_image = tf.image.crop_to_bounding_box(img, 310, z, 1560, 310)

    with tf.Session() as sess2:
      cropped_image_ = cropped_image.eval()

    plt.figure(1)
    plt.imshow(cropped_image_)
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(3.1 / 3, 15.6 / 3)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # 保存分割后的图像
    plt.savefig("./segment/%s_%s.jpg" % (str(cycle), str(count)), dpi=96)
    # plt.show()

    # 使用分割后的图像进行识别检测
    image_data = tf.gfile.FastGFile("./segment/%s_%s.jpg" % (str(cycle), str(count)), 'rb').read()

    with tf.Session() as sess:
        # 根据graph对图片进行初步预测
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score > 0.5 and human_string == "valid component":
                output_data_arr[count-1] = 1
            elif score > 0.5 and human_string == "invalid component":
                output_data_arr[count-1] = 0

    count = count + 1

  i = 0

  flag = 1

  # 创建输出文件
  fw = open(r"./output/result.txt", 'a')

  output = ""

  # 输出数据处理
  while i < 3:
      if output_data_arr[i] == 0:
          if flag == 0:
              output = output + ","
          flag = 0
          output = output + str(i + 1)
      i = i + 1

  if flag == 1:
      output = "合格"
  elif flag == 0:
      output = "不合格 " + output

  output = str(cycle) + " " + output
  # 数据输出到文件
  print(output, file=fw)

  cycle = cycle + 1