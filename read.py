import mpu


sample = mpu.io.read("logs/sample.pickle")
pbconcat = mpu.io.read("logs/pbcon.pickle")
test = mpu.io.read("logs/samplewithouthotencoding.pickle")
nohec = mpu.io.read("logs/nohec.pickle")
z = 0