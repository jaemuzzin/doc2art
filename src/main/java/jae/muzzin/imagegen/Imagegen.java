package jae.muzzin.imagegen;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.DeConv2DConfig;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Pooling2DConfig;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.weightinit.impl.XavierInitScheme;

/**
 *
 * @author Admin
 */
public class Imagegen {

    public static void main(String[] args) throws IOException {

        SameDiff sd = SameDiff.create();
        if (new File("gan.model").exists()) {
            sd = SameDiff.load(new File("gan.model"), false);
            //print gen example
            sd.getVariable("generator_input").setArray(Nd4j.rand(DataType.FLOAT, 1, 8, 5, 5));
            var exampleGenImage = sd.math.step(sd.getVariable("generator"), .5).eval().reshape(28, 28);
            System.err.println(exampleGenImage.toStringFull().replaceAll(" ", "").replaceAll("1", "*").replaceAll("0", " ").replaceAll(",", ""));

            System.exit(0);
        }

        autoencoder(sd);

        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                //.l2(1e-4) //L2 regularization
                .updater(new Nadam(learningRate)) //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("input") //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label") //DataSet label array should be associated with variable "label"
                .build();

        // You can add validation evaluations as well, but they have some issues in beta5 and most likely won't work.
        // If you want to use them, use the SNAPSHOT build.
        sd.setTrainingConfig(config);

        // Adding a listener to the SameDiff instance is necessary because of a beta5 bug, and is not necessary in snapshots
        sd.addListeners(new ScoreListener(20));

        int batchSize = 512;

        if (!new File("autoencoder.model").exists()) {
            for (int i = 0; i < 120; i++) {
                DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);
                System.err.println("training..");
                while (trainData.hasNext()) {
                    DataSet ds = trainData.next();
                    var realDs = new DataSet(ds.getFeatures(), ds.getFeatures());
                    sd.fit(realDs);
                    System.err.print(".");
                    System.err.flush();
                }
                System.err.println("Done.");
                DataSetIterator testData = new MnistDataSetIterator(batchSize, false, 12345);
                System.err.println("testing..");
                RegressionEvaluation evaluation = new RegressionEvaluation();
                INDArray exampleRow = null;
                INDArray exampleImage = null;
                while (testData.hasNext()) {
                    DataSet ds = testData.next();
                    var realDs = new DataSet(ds.getFeatures(), ds.getFeatures());
                    sd.evaluate(new ViewIterator(realDs, Math.min(batchSize, ds.numExamples() - 1)), "out", evaluation);
                    exampleRow = ds.getFeatures().getRow(0);
                }
                //print in
                //var imageOutput = exampleRow.reshape(-1, 28, 28);
                //System.err.println(imageOutput.toStringFull());
                //print out
                sd.getVariable("input").setArray(Nd4j.expandDims(exampleRow, 0));
                var imageOutput = sd.math.step(sd.getVariable("out"), .1).eval().reshape(-1, 28, 28);
                System.err.println(imageOutput.toStringFull().replaceAll(" ", "").replaceAll("1", "*").replaceAll("0", " ").replaceAll(",", ""));
                System.err.println(evaluation.averageMeanSquaredError());
                sd.save(new File("autoencoder.model"), true);
            }
        }
        sd = SameDiff.load(new File("autoencoder.model"), false);

        //read batch of real examples, encode them, label as 0
        var decoder_input = sd.placeHolder("decoder_input", DataType.FLOAT, -1, 8, 5, 5);
        var generator_input = sd.placeHolder("generator_input", DataType.FLOAT, -1, 8, 5, 5);
        var gan_label = sd.placeHolder("gan_label", DataType.FLOAT, -1, 1);
        var generator = generator(sd, "generator", generator_input, 28);
        var disc_input = sd.placeHolder("disc_input", DataType.FLOAT, -1, 200);
        //add input to generator output, will need to zero out the other depending
        var decoder = decoder(sd, "decoder", decoder_input, 28);
        var encoder = encoder(sd, generator, 28);
        var disc = discriminator(sd, encoder, "disc");//used to train generator
        var discOfData = discriminatorOfData(sd, disc_input, "disc_of_data");//the one actually trained
        var generator_loss = genLoss(sd, "generator_loss", disc, gan_label);
        var disc_loss = discLoss(sd, "disc_loss", discOfData, gan_label);
        for (int i = 0; i < 200000; i++) {
            DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);
            Evaluation evaluation = new Evaluation();

            //setup training
            sd.setLossVariables(disc_loss);
            sd.convertToVariables(Arrays.asList(new SDVariable[]{sd.getVariable("disc_w0"), sd.getVariable("disc_w1"), sd.getVariable("disc_b0"), sd.getVariable("disc_b1")}));

            System.err.println("Training GAN...");
            boolean first = true;
            var regEvalDisc = new RegressionEvaluation();
            while (first || trainData.hasNext() && (evaluation.falseNegatives().get(1)>0 || evaluation.falsePositives().get(1) > 0)) {
                first = false;
                evaluation = new Evaluation();
                DataSet ds = trainData.next();
                sd.getVariable("input").setArray(ds.getFeatures());
                var realTrainingFeatures = sd.getVariable("flat_hidden").eval();//encode teh real images
                var realTrainingLables = Nd4j.zeros(DataType.FLOAT, ds.getFeatures().shape()[0], 1);
                var fakeTrainingLables = Nd4j.ones(DataType.FLOAT, ds.getFeatures().shape()[0], 1);

                //generate same number of fakes, label as 1
                sd.getVariable("generator_input").setArray(Nd4j.rand(DataType.FLOAT, ds.getFeatures().shape()[0], 8, 5, 5));
                var fakeTrainingImages = generator.eval();
                sd.getVariable("input").setArray(fakeTrainingImages);
                var fakeTrainingFeatures = sd.getVariable("flat_hidden").eval();//encode teh real images
                TrainingConfig discConfig = new TrainingConfig.Builder()
                        .l2(1e-4) //L2 regularization
                        .updater(new Nadam(1e-5)) //Adam optimizer with specified learning rate
                        .dataSetFeatureMapping("disc_input") //DataSet features array should be associated with variable "input"
                        .dataSetLabelMapping("gan_label") //DataSet label array should be associated with variable "label"
                        .build();
                
                sd.setTrainingConfig(discConfig);
                var myDs = new DataSet(Nd4j.concat(0, realTrainingFeatures, fakeTrainingFeatures), Nd4j.concat(0, realTrainingLables, fakeTrainingLables));
                sd.fit(myDs);
                sd.evaluate(new ViewIterator(myDs, Math.min(batchSize, myDs.numExamples() - 1)), "disc_of_data", evaluation);
                sd.evaluate(new ViewIterator(myDs, Math.min(batchSize, myDs.numExamples() - 1)), "disc_of_data", regEvalDisc);
            }
            if (!trainData.hasNext()) {
                System.err.println("Exited GAN training without success.");
                System.err.println(regEvalDisc.averageMeanAbsoluteError());
                continue;
            }
            System.err.println(evaluation.confusionMatrix());
            //Pretrain the generator
            var fakeGenTrainingLables = Nd4j.ones(DataType.FLOAT, batchSize, 1);
            double genlearningRate = 1e-4;
            TrainingConfig genConfig = new TrainingConfig.Builder()
                    .l2(1e-4) //L2 regularization
                    .updater(new Nadam(genlearningRate)) //Adam optimizer with specified learning rate
                    .dataSetFeatureMapping("generator_input") //DataSet features array should be associated with variable "input"
                    .dataSetLabelMapping("gan_label") //DataSet label array should be associated with variable "label"
                    .build();
            sd.setTrainingConfig(genConfig);
            sd.setLossVariables(generator_loss);
            sd.convertToConstants(Arrays.asList(new SDVariable[]{sd.getVariable("w0"), sd.getVariable("w1"), sd.getVariable("b0"), sd.getVariable("b1"), sd.getVariable("disc_w0"), sd.getVariable("disc_w1"), sd.getVariable("disc_b0"), sd.getVariable("disc_b1")}));
            
            System.err.println("Training GEN...");
            var regEval = new RegressionEvaluation();
            for (int e = 0; e < 1 || evaluation.falseNegatives().get(1) < evaluation.truePositives().get(1); e++) {
                evaluation = new Evaluation();
                DataSet gends = new DataSet(Nd4j.rand(DataType.FLOAT, batchSize, 8, 5, 5), fakeGenTrainingLables);
                sd.fit(gends);
                sd.evaluate(new ViewIterator(gends, Math.min(batchSize, gends.numExamples() - 1)), "disc", evaluation);
                sd.evaluate(new ViewIterator(gends, Math.min(batchSize, gends.numExamples() - 1)), "disc", regEval);
                if (e % 10 == 0) {
                    sd.getVariable("generator_input").setArray(Nd4j.rand(DataType.FLOAT, 1, 8, 5, 5));
                    System.err.println(evaluation.confusionMatrix());
                    var imageOutput = sd.math.step(generator, 0.5).eval().reshape(1, 28, 28);
                    System.err.println(imageOutput.toStringFull().replaceAll(" ", "").replaceAll("1", "*").replaceAll("0", " ").replaceAll(",", ""));
                    System.err.println("This should get higher:" + regEval.averageMeanAbsoluteError());
                    regEval = new RegressionEvaluation();
                }
            }
            System.err.println(evaluation.confusionMatrix());
            //print gen example
            sd.getVariable("generator_input").setArray(Nd4j.rand(DataType.FLOAT, 1, 8, 5, 5));
            var imageOutput = sd.math.step(generator, 0.5).eval().reshape(1, 28, 28);
            System.err.println(imageOutput.toStringFull().replaceAll(" ", "").replaceAll("1", "*").replaceAll("0", " ").replaceAll(",", ""));

            sd.save(new File("gan.model"), true);
        }
    }

    public static SDVariable generator(SameDiff sd, String varName, SDVariable in, int width) {
        SDVariable dw0 = sd.var("gw0", new XavierInitScheme('c', 5 * 5 * 8, 10 * 10 * 4), DataType.FLOAT, 2, 2, 4, 8);
        SDVariable db0 = sd.zero("gb0", DataType.FLOAT, 4);
        SDVariable deconv1 = sd.nn().tanh(sd.cnn().deconv2d(in, dw0, db0, DeConv2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build()));
        SDVariable dw1 = sd.var("gw1", new XavierInitScheme('c', 10 * 10 * 4, 12 * 12 * 2), DataType.FLOAT, 3, 3, 2, 4);
        SDVariable db1 = sd.zero("gb1", DataType.FLOAT, 2);
        SDVariable deconv2 = sd.nn().tanh(sd.cnn().deconv2d(deconv1, dw1, db1, DeConv2DConfig.builder().kH(3).kW(3).sH(1).sW(1).build()));
        SDVariable dw2 = sd.var("gw2", new XavierInitScheme('c', 12 * 12 * 2, 24 * 24 * 2), DataType.FLOAT, 2, 2, 2, 2);
        SDVariable db2 = sd.zero("gb2", DataType.FLOAT, 2);
        SDVariable deconv3 = sd.nn().tanh(sd.cnn().deconv2d(deconv2, dw2, db2, DeConv2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build()));
        SDVariable dw3 = sd.var("gw3", new XavierInitScheme('c', 24 * 24 * 2, 26 * 26 * 2), DataType.FLOAT, 3, 3, 2, 2);
        SDVariable db3 = sd.zero("gb3", DataType.FLOAT, 2);
        SDVariable deconv4 = sd.nn().tanh(sd.cnn().deconv2d(deconv3, dw3, db3, DeConv2DConfig.builder().kH(3).kW(3).sH(1).sW(1).build()));
        SDVariable dw4 = sd.var("gw4", new XavierInitScheme('c', 26 * 26 * 2, 28 * 28 * 1), DataType.FLOAT, 3, 3, 1, 2);
        SDVariable db4 = sd.zero("gb4", DataType.FLOAT, 1);
        SDVariable deconv5 = sd.nn().sigmoid(sd.cnn().deconv2d(deconv4, dw4, db4, DeConv2DConfig.builder().kH(3).kW(3).sH(1).sW(1).build()));
        var out = deconv5.reshape("generator", sd.constant(Nd4j.create(new int[][]{{-1, width * width}})));
        return out;
    }

    public static SDVariable decoder(SameDiff sd, String varName, SDVariable in, int width) {
        SDVariable deconv1 = sd.nn().relu(sd.cnn().deconv2d(in, sd.getVariable("dw0"), sd.getVariable("db0"), DeConv2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build()), 0);
        SDVariable deconv2 = sd.nn().relu(sd.cnn().deconv2d(deconv1, sd.getVariable("dw1"), sd.getVariable("db1"), DeConv2DConfig.builder().kH(3).kW(3).sH(1).sW(1).build()), 0);
        SDVariable deconv3 = sd.nn().relu(sd.cnn().deconv2d(deconv2, sd.getVariable("dw2"), sd.getVariable("db2"), DeConv2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build()), 0);
        SDVariable deconv4 = sd.nn().relu(sd.cnn().deconv2d(deconv3, sd.getVariable("dw3"), sd.getVariable("db3"), DeConv2DConfig.builder().kH(3).kW(3).sH(1).sW(1).build()), 0);
        SDVariable deconv5 = sd.nn().hardSigmoid(sd.cnn().deconv2d(deconv4, sd.getVariable("dw4"), sd.getVariable("db4"), DeConv2DConfig.builder().kH(3).kW(3).sH(1).sW(1).build()));

        return deconv5.reshape(varName, sd.constant(Nd4j.create(new int[][]{{-1, width * width}})));
    }

    /**
     * L = -log(sigmoid(D(G(z))))
  This is the trick used in the original paper to avoid vanishing gradients
  early in training. See `Generative Adversarial Nets`
  (https://arxiv.org/abs/1406.2661) 
     * @return 
     */
    public static SDVariable genLoss(SameDiff sd, String varName, SDVariable disc, SDVariable label) {
        return sd.math.log(disc);
    }

    public static SDVariable discLoss(SameDiff sd, String varName, SDVariable descrim, SDVariable label) {
        return sd.loss.logLoss(varName, label, descrim);
    }

    public static SDVariable discriminator(SameDiff sd, SDVariable in, String varName) {
        var w0 = sd.var("disc_w0", new XavierInitScheme('c', 200, 20), DataType.FLOAT, 200, 20);
        var b0 = sd.zero("disc_b0", DataType.FLOAT, 1, 20);
        var w1 = sd.var("disc_w1", new XavierInitScheme('c', 20, 1), DataType.FLOAT, 20, 1);
        var b1 = sd.zero("disc_b1", DataType.FLOAT, 1, 1);
        return sd.nn.sigmoid(varName, sd.nn.relu(in.mmul(w0).add(b0), 0).mmul(w1).add(b1));
    }

    public static SDVariable discriminatorOfData(SameDiff sd, SDVariable in, String varName) {
        return sd.nn.sigmoid(varName, sd.nn.relu(in.mmul(sd.getVariable("disc_w0")).add(sd.getVariable("disc_b0")), 0).mmul(sd.getVariable("disc_w1")).add(sd.getVariable("disc_b1")));
    }

    /**
     *
     * @param sd
     * @param in shape -1*w*w
     * @return
     */
    public static SDVariable encoder(SameDiff sd, SDVariable in, int w) {
        int c = 16;
        int d = 32;
        int nIn = w * w;

        //unflatten
        SDVariable reshaped = in.reshape(-1, 1, w, w);

        Pooling2DConfig poolConfig = Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build();

        Conv2DConfig convConfig = Conv2DConfig.builder().kH(3).kW(3).build();

        //26x26x4
        SDVariable conv1 = sd.cnn().conv2d(reshaped, sd.getVariable("w0"), sd.getVariable("b0"), convConfig);

        //13x13x4
        // layer 2: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool1 = sd.cnn().maxPooling2d(conv1, poolConfig);

        SDVariable relu1 = sd.nn().relu(pool1, 0);
        //11x11x8
        SDVariable conv2 = sd.cnn().conv2d(relu1, sd.getVariable("w1"), sd.getVariable("b1"), convConfig);

        //5x5x8
        SDVariable pool2 = sd.cnn().maxPooling2d(conv2, poolConfig);

        return sd.reshape(sd.nn().relu("encoder", pool2, 0), -1, 5 * 5 * 8);
    }

    public static void autoencoder(SameDiff sd) {

        //Properties for MNIST dataset:
        int c = 16;
        int d = 32;
        int w = 28;
        int nIn = w * w;

        //Create input and label variables
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, nIn);
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nIn);

        //unflatten
        SDVariable reshaped = in.reshape(-1, 1, w, w);

        Pooling2DConfig poolConfig = Pooling2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build();

        Conv2DConfig convConfig = Conv2DConfig.builder().kH(3).kW(3).build();

        // layer 1: Conv2D with a 3x3 kernel and 4 output channels
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', w * w, 26 * 26 * 4), DataType.FLOAT, 3, 3, 1, 4);
        SDVariable b0 = sd.zero("b0", 4);

        //26x26x4
        SDVariable conv1 = sd.cnn().conv2d(reshaped, w0, b0, convConfig);

        //13x13x4
        // layer 2: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool1 = sd.cnn().maxPooling2d(conv1, poolConfig);

        SDVariable relu1 = sd.nn().relu(pool1, 0);

        // layer 3: Conv2D with a 3x3 kernel and 8 output channels
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', 13 * 13 * 4, 11 * 11 * 8), DataType.FLOAT, 3, 3, 4, 8);
        SDVariable b1 = sd.zero("b1", 8);

        //11x11x8
        SDVariable conv2 = sd.cnn().conv2d(relu1, w1, b1, convConfig);

        //5x5x8
        // layer 4: MaxPooling2D with a 2x2 kernel and stride, and ReLU activation
        SDVariable pool2 = sd.cnn().maxPooling2d(conv2, poolConfig);

        SDVariable relu2 = sd.nn().relu("encoder_output", pool2, 0);

        //200 long
        relu2.reshape("flat_hidden", sd.constant(Nd4j.create(new int[][]{{-1, 5 * 5 * 8}})));

        //W,H,OUT,IN
        SDVariable dw0 = sd.var("dw0", new XavierInitScheme('c', 5 * 5 * 8, 10 * 10 * 4), DataType.FLOAT, 2, 2, 4, 8);
        SDVariable db0 = sd.zero("db0", 4);
        SDVariable deconv1 = sd.nn().relu(sd.cnn().deconv2d(relu2, dw0, db0, DeConv2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build()), 0);
        SDVariable dw1 = sd.var("dw1", new XavierInitScheme('c', 10 * 10 * 4, 12 * 12 * 2), DataType.FLOAT, 3, 3, 2, 4);
        SDVariable db1 = sd.zero("db1", 2);
        SDVariable deconv2 = sd.nn().relu(sd.cnn().deconv2d(deconv1, dw1, db1, DeConv2DConfig.builder().kH(3).kW(3).sH(1).sW(1).build()), 0);
        SDVariable dw2 = sd.var("dw2", new XavierInitScheme('c', 12 * 12 * 2, 24 * 24 * 2), DataType.FLOAT, 2, 2, 2, 2);
        SDVariable db2 = sd.zero("db2", 2);
        SDVariable deconv3 = sd.nn().relu(sd.cnn().deconv2d(deconv2, dw2, db2, DeConv2DConfig.builder().kH(2).kW(2).sH(2).sW(2).build()), 0);
        SDVariable dw3 = sd.var("dw3", new XavierInitScheme('c', 24 * 24 * 2, 26 * 26 * 2), DataType.FLOAT, 3, 3, 2, 2);
        SDVariable db3 = sd.zero("db3", 2);
        SDVariable deconv4 = sd.nn().relu(sd.cnn().deconv2d(deconv3, dw3, db3, DeConv2DConfig.builder().kH(3).kW(3).sH(1).sW(1).build()), 0);
        SDVariable dw4 = sd.var("dw4", new XavierInitScheme('c', 26 * 26 * 2, 28 * 28 * 1), DataType.FLOAT, 3, 3, 1, 2);
        SDVariable db4 = sd.zero("db4", 1);
        SDVariable deconv5 = sd.nn().hardSigmoid(sd.cnn().deconv2d(deconv4, dw4, db4, DeConv2DConfig.builder().kH(3).kW(3).sH(1).sW(1).build()));

        var out = deconv5.reshape("out", sd.constant(Nd4j.create(new int[][]{{-1, w * w}})));
        SDVariable loss = sd.loss().meanSquaredError("loss", label, out, null);
        sd.setLossVariables(loss);
    }
}
