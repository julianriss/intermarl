import mpu

postprocessed_batch = mpu.io.read("/Users/julian/Desktop/postprocessed_batch.pickle")
concat = mpu.io.read("/Users/julian/Desktop/concatenated.pickle")
rw = mpu.io.read("/Users/julian/Desktop/rw.pickle")
sampelbatch =  mpu.io.read("/Users/julian/Desktop/sampledbatch.pickle")

zero = mpu.io.read("logs/sample0.pickle")
one = mpu.io.read("logs/sample1.pickle")
two = mpu.io.read("logs/sample2.pickle")
three = mpu.io.read("logs/sample3.pickle")

z = 0