# RobotDrawImage

## 说明
目前采用jupyter调用摄像头，拍摄照片，并采用QMUPD生成素描图像。之后，编写了一套算法，将素描图像处理成多个线条，并将线条生成为机器人运行的点位。

## 经常遇到的问题
1. jupyter调用摄像头时候无法打开
使用chrome浏览器，需要在启动命令中，加入unsafely的选项，例如，ip要替换为服务器的地址
"C:\Program Files\Google\Chrome\Application\chrome.exe" --unsafely-treat-insecure-origin-as-secure="http://192.168.1.152:8888/"