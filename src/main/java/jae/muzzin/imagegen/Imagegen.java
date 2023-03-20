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

        int batchSize = 64;

        if (!new File("autoencoder.model").exists()) {
            for (int i = 0; i < 20; i++) {
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
                while (testData.hasNext()) {
                    DataSet ds = testData.next();
                    var realDs = new DataSet(ds.getFeatures(), ds.getFeatures());
                    sd.evaluate(new ViewIterator(realDs, Math.min(batchSize, ds.numExamples() - 1)), "out", evaluation);
                }

                System.err.println(evaluation.averageMeanSquaredError());
            }
            sd.save(new File("autoencoder.model"), true);
        } else {
            sd = SameDiff.load(new File("autoencoder.model"), false);
        }

        //read batch of real examples, encode them, label as 0
        var generator_input = sd.placeHolder("generator_input", DataType.FLOAT, -1, 1);
        var gan_label = sd.placeHolder("gan_label", DataType.FLOAT, -1, 1);
        var generator = generator(sd, "generator", generator_input);
        var disc_input = sd.placeHolder("disc_input", DataType.FLOAT, -1, 200);
        //add input to generator output, will need to zero out the other depending
        var disc = discriminator(sd, generator, "disc");
        var discOfData = discriminatorOfData(sd, disc_input, "disc_of_data");
        var generator_loss = genLoss(sd, "generator_loss", disc, gan_label);
        var disc_loss = discLoss(sd, "disc_loss", discOfData, gan_label);
        for (int i = 0; i < 20; i++) {
            DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);

            while (trainData.hasNext()) {
                System.err.println("Training GAN...");

                DataSet ds = trainData.next();
                //Pretrain the generator
                var fakeGenTrainingLables = Nd4j.zeros(ds.getFeatures().shape()[0], 1);
                double genlearningRate = 1e-3;
                TrainingConfig genConfig = new TrainingConfig.Builder()
                        //.l2(1e-4) //L2 regularization
                        .updater(new Nadam(learningRate)) //Adam optimizer with specified learning rate
                        .dataSetFeatureMapping("generator_input") //DataSet features array should be associated with variable "input"
                        .dataSetLabelMapping("gan_label") //DataSet label array should be associated with variable "label"
                        .build();
                sd.setTrainingConfig(genConfig);
                sd.setLossVariables(generator_loss);
                sd.fit(new DataSet(Nd4j.rand(DataType.FLOAT, ds.getFeatures().shape()[0], 1), fakeGenTrainingLables));

                sd.getVariable("input").setArray(ds.getFeatures());
                var realTrainingFeatures = sd.getVariable("flat_hidden").eval();
                var realTrainingLables = Nd4j.zeros(ds.getFeatures().shape()[0], 1);
                var fakeTrainingLables = Nd4j.ones(ds.getFeatures().shape()[0], 1);

                //generate same number of fakes, label as 1
                sd.getVariable("generator_input").setArray(Nd4j.rand(DataType.FLOAT, ds.getFeatures().shape()[0], 1));
                var fakeTrainingFeatures = generator.eval();
                sd.setLossVariables(disc_loss);
                TrainingConfig discConfig = new TrainingConfig.Builder()
                        //.l2(1e-4) //L2 regularization
                        .updater(new Nadam(learningRate)) //Adam optimizer with specified learning rate
                        .dataSetFeatureMapping("disc_input") //DataSet features array should be associated with variable "input"
                        .dataSetLabelMapping("gan_label") //DataSet label array should be associated with variable "label"
                        .build();
                sd.setTrainingConfig(discConfig);
                var myDs = new DataSet(Nd4j.concat(0, realTrainingFeatures, fakeTrainingFeatures), Nd4j.concat(0, realTrainingLables, fakeTrainingLables));
                var h = sd.fit(myDs);
                Evaluation evaluation = new Evaluation();
                sd.evaluate(new ViewIterator(myDs, Math.min(batchSize, myDs.numExamples() - 1)), "disc_of_data", evaluation);
                System.err.println(evaluation.confusionMatrix());
            }
        }
    }

    public static SDVariable generator(SameDiff sd, String varName, SDVariable in) {
        var w0 = sd.var("gen_w0", new XavierInitScheme('c', 1, 50), DataType.FLOAT, 1, 50);
        var w1 = sd.var("gen_w1", new XavierInitScheme('c', 50, 200), DataType.FLOAT, 50, 200);
        var b0 = sd.zero("gen_b0", DataType.FLOAT, 1, 50);
        var b1 = sd.zero("gen_b1", DataType.FLOAT, 1, 200);
        return sd.nn.sigmoid(varName, sd.nn.relu(sd.nn.relu(in.mmul(w0).add(b0), 0).mmul(w1).add(b1), 0));
    }

    public static SDVariable genLoss(SameDiff sd, String varName, SDVariable disc, SDVariable label) {
        return sd.loss.absoluteDifference(varName, label, disc, null);
    }

    public static SDVariable discLoss(SameDiff sd, String varName, SDVariable descrim, SDVariable label) {
        return sd.loss.absoluteDifference(label, descrim, null);
    }

    public static SDVariable discriminator(SameDiff sd, SDVariable in, String varName) {
        var w0 = sd.var("disc_w0", new XavierInitScheme('c', 200, 20), DataType.FLOAT, 200, 20);
        var b0 = sd.zero("disc_b0", DataType.FLOAT, 1, 20);
        var w1 = sd.var("disc_w1", new XavierInitScheme('c', 20, 1), DataType.FLOAT, 20, 1);
        var b1 = sd.zero("disc_b1", DataType.FLOAT, 1, 1);
        return sd.nn.hardSigmoid(varName, sd.nn.relu(in.mmul(w0).add(b0), 0).mmul(w1).add(b1));
    }

    public static SDVariable discriminatorOfData(SameDiff sd, SDVariable in, String varName) {
        return sd.nn.sigmoid(varName, sd.nn.relu(in.mmul(sd.getVariable("disc_w0")).add(sd.getVariable("disc_b0")), 0).mmul(sd.getVariable("disc_w1")).add(sd.getVariable("disc_b1")));
    }

    public static void autoencoder(SameDiff sd) {

        //Properties for MNIST dataset:
        int c = 16;
        int d = 32;
        int w = 28;
        int nIn = w * w;

        //Create input and label variables
        SDVariable in = sd.placeHolder("input", DataType.FLOAT, -1, nIn);                 //Shape: [?, 784] - i.e., minibatch x 784 for MNIST
        SDVariable label = sd.placeHolder("label", DataType.FLOAT, -1, nIn);             //Shape: [?, 10] - i.e., minibatch x 10 for MNIST

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

        SDVariable relu2 = sd.nn().relu(pool2, 0);

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
