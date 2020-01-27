package ru.spbu.apmath;

import java.awt.image.*;
import java.io.*;
import java.util.*;
import java.util.List;

import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.preprocessor.FeedForwardToCnnPreProcessor;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.*;
import org.deeplearning4j.util.ModelSerializer;

import org.nd4j.linalg.activations.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.*;
import org.nd4j.linalg.primitives.ImmutableTriple;

import javax.imageio.ImageIO;

public class GAN
{
    private final int _randomSeed = 3455;
    private final int _printEvery = 10;
    private final int _saveEvery = 100;
    private final double _dis_learning_rate = 2e-3;
    private final double _frozen_learning_rate = 0.0;
    private final double _gen_learning_rate = 4e-3;
    private final String _resPath = ".\\src\\main\\java\\ru\\spbu\\apmath\\";
    private final int _trainingStartBatchNumber = 0;

    private final String _dataSetName;
    private final int _width;
    private final int _height;
    private final int _channels;
    private final int _flatSize;
    private final int _genInputCount;
    private final int _numTrainingSamples;
    private final int _numTestSamples;
    private final int _batchSize;
    private final int _numIterations;
    private final int _numGenSamples;

    private final ComputationGraph _dis;
    private final ComputationGraph _gan;
    private final ComputationGraph _gen;

    public static void main(String[] args) throws Exception
    {
        // Создаём модель для генерации чисел на основе MNIST.
        GAN model = new GAN("mnist", false);

        // Производим предобработку данных и обучаем модель.
        DataSetIterator dataset = model.prepareData();
        model.train(dataset);

        // Генерируем и сохраняем изображение.
        model.create("GeneratedImage.png");
    }

    private ImmutableTriple<ComputationGraph, ComputationGraph, ComputationGraph> _createNetworks()
    {
        // Создаём три сети:

        // Обучаемый дискриминатор.
        ComputationGraph dis = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(_randomSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .l2(0.0001)
                // Функция активации всех свёрточных слоёв и
                // первого полносвязного - гиперболический тангенс.
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                // Слои:
                // 0. Трёхмерный вход размерности [_width, _height, _channels].
                .addInputs("dis_input")
                .setInputTypes(InputType.convolutionalFlat(_width, _height, _channels))
                // 1. Батч-нормализация.
                .addLayer("dis_batch_norm", new BatchNormalization.Builder()
                        .updater(new RmsProp(_dis_learning_rate, 1e-8, 1e-8))
                        .build(), "dis_input")
                // 2. Свёрточный слой (ядро 5x5...
                .addLayer("dis_conv_1", new ConvolutionLayer.Builder(5, 5)
                        // ...шаг 2...
                        .stride(2, 2)
                        .updater(new RmsProp(_dis_learning_rate, 1e-8, 1e-8))
                        // ..._channels каналов -> 64 канала).
                        .nIn(_channels)
                        .nOut(64)
                        .build(), "dis_batch_norm")
                // 3. Субдискретизирующий слой (maxpool...
                .addLayer("dis_maxpool_1", new SubsamplingLayer.Builder(PoolingType.MAX)
                        //...ядро 2x2...
                        .kernelSize(2, 2)
                        //...шаг 1).
                        .stride(1, 1)
                        .build(), "dis_conv_1")
                // 4. Свёрточный слой (ядро 5x5...
                .addLayer("dis_conv_2", new ConvolutionLayer.Builder(5, 5)
                        // ...шаг 2...
                        .stride(2, 2)
                        .updater(new RmsProp(_dis_learning_rate, 1e-8, 1e-8))
                        // ...64 канала -> 128 каналов).
                        .nIn(64)
                        .nOut(128)
                        .build(), "dis_maxpool_1")
                // 5. Субдискретизирующий слой (maxpool...
                .addLayer("dis_maxpool_2", new SubsamplingLayer.Builder(PoolingType.MAX)
                        //...ядро 2x2...
                        .kernelSize(2, 2)
                        //...шаг 1).
                        .stride(1, 1)
                        .build(), "dis_conv_2")
                // 6. Полносвязный слой.
                .addLayer("dis_dense", new DenseLayer.Builder()
                        .updater(new RmsProp(_dis_learning_rate, 1e-8, 1e-8))
                        // Размер - 1024 нейрона.
                        .nOut(1024)
                        .build(), "dis_maxpool_2")
                // 7. Выходной полносвязный слой.
                .addLayer("dis_output", new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .updater(new RmsProp(_dis_learning_rate, 1e-8, 1e-8))
                        // Размер - один нейрон для бинарной классификации.
                        .nOut(1)
                        // Функция активации - сигмоида.
                        .activation(Activation.SIGMOID)
                        .build(), "dis_dense")
                .setOutputs("dis_output")
                .build());
        dis.init();
        // Выводим сводное описание дискриминатора.
        System.out.println(dis.summary());

        // Необучаемый генератор.
        ComputationGraph gen = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(_randomSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .l2(0.0001)
                // Функция активации всех полносвязных и всех (кроме последнего)
                // свёрточных слоёв  - гиперболический тангенс.
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .graphBuilder()
                // Слои:
                // 0. Линейный вход размерности _genInputCount.
                .addInputs("gen_input")
                .setInputTypes(InputType.feedForward(_genInputCount))
                // 1. Батч-нормализация.
                .addLayer("gen_batch_norm_1", new BatchNormalization.Builder()
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        .build(), "gen_input")
                // 2. Полносвязный слой.
                .addLayer("gen_dense_1", new DenseLayer.Builder()
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        // Размер - 1024 нейрона.
                        .nOut(1024)
                        .build(), "gen_batch_norm_1")
                // 3. Полносвязный слой.
                .addLayer("gen_dense_2", new DenseLayer.Builder()
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        // Размер - 7 * 7 * 128 нейронов.
                        .nOut(7 * 7 * 128)
                        .build(), "gen_dense_1")
                // 4. Батч-нормализация.
                .addLayer("gen_batch_norm_2", new BatchNormalization.Builder()
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        .build(), "gen_dense_2")
                // Преобразовываем линейный выход из 7 * 7 * 128 нейронов в трёхмерный размерности [7, 7, 128].
                .inputPreProcessor("gen_upsampling_1", new FeedForwardToCnnPreProcessor(7, 7, 128))
                // 5. Upsampling-слой (ядро 2x2).
                .addLayer("gen_upsampling_1", new Upsampling2D.Builder(2)
                        .build(), "gen_batch_norm_2")
                // 6. Свёрточный слой (ядро 5x5...
                .addLayer("gen_conv_1", new ConvolutionLayer.Builder(5, 5)
                        // ...шаг 1...
                        .stride(1, 1)
                        // ...выравнивание 2x2...
                        .padding(2, 2)
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        // ...128 каналов -> 64 канала.
                        .nIn(128)
                        .nOut(64)
                        .build(), "gen_upsampling_1")
                // 7. Upsampling-слой (ядро 2x2).
                .addLayer("gen_upsampling_2", new Upsampling2D.Builder(2)
                        .build(), "gen_conv_1")
                // 8. Свёрточный слой (ядро 5x5...
                .addLayer("gen_conv_2", new ConvolutionLayer.Builder(5, 5)
                        // ...шаг 1...
                        .stride(1, 1)
                        // ...выравнивание 2x2...
                        .padding(2, 2)
                        // ...функция активации - сигмоида...
                        .activation(Activation.SIGMOID)
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        // ...64 канала -> _channels каналов.
                        .nIn(64)
                        .nOut(_channels)
                        .build(), "gen_upsampling_2")
                .setOutputs("gen_conv_2")
                .build());
        gen.init();
        // Выводим сводное описание генератора.
        System.out.println(gen.summary());

        // Обучаемый генератор с необучаемым дискриминатором.
        ComputationGraph gan = new ComputationGraph(new NeuralNetConfiguration.Builder()
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .inferenceWorkspaceMode(WorkspaceMode.ENABLED)
                .seed(_randomSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .activation(Activation.TANH)
                .weightInit(WeightInit.XAVIER)
                .l2(0.0001)
                .graphBuilder()
                // Полная копия генератора (кроме скорости обучения).
                .addInputs("gan_input")
                .setInputTypes(InputType.feedForward(_genInputCount))
                .addLayer("gan_batch_norm_1", new BatchNormalization.Builder()
                        .updater(new RmsProp(_gen_learning_rate, 1e-8, 1e-8))
                        .build(), "gan_input")
                .addLayer("gan_dense_1", new DenseLayer.Builder()
                        .updater(new RmsProp(_gen_learning_rate, 1e-8, 1e-8))
                        .nOut(1024)
                        .build(), "gan_batch_norm_1")
                .addLayer("gan_dense_2", new DenseLayer.Builder()
                        .updater(new RmsProp(_gen_learning_rate, 1e-8, 1e-8))
                        .nOut(7 * 7 * 128)
                        .build(), "gan_dense_1")
                .addLayer("gan_batch_norm_2", new BatchNormalization.Builder()
                        .updater(new RmsProp(_gen_learning_rate, 1e-8, 1e-8))
                        .build(), "gan_dense_2")
                .inputPreProcessor("gan_upsampling_1", new FeedForwardToCnnPreProcessor(7, 7, 128))
                .addLayer("gan_upsampling_1", new Upsampling2D.Builder(2)
                        .build(), "gan_batch_norm_2")
                .addLayer("gan_conv_1", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .padding(2, 2)
                        .updater(new RmsProp(_gen_learning_rate, 1e-8, 1e-8))
                        .nIn(128)
                        .nOut(64)
                        .build(), "gan_upsampling_1")
                .addLayer("gan_upsampling_2", new Upsampling2D.Builder(2)
                        .build(), "gan_conv_1")
                .addLayer("gan_conv_2", new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        .padding(2, 2)
                        .activation(Activation.SIGMOID)
                        .updater(new RmsProp(_gen_learning_rate, 1e-8, 1e-8))
                        .nIn(64)
                        .nOut(_channels)
                        .build(), "gan_upsampling_2")

                // Полная копия дискриминатора (кроме скорости обучения).
                .addLayer("gan_batch_norm_d", new BatchNormalization.Builder()
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        .build(), "gan_conv_2")
                .addLayer("gan_conv_1_d", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        .nIn(_channels)
                        .nOut(64)
                        .build(), "gan_batch_norm_d")
                .addLayer("gan_maxpool_1", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(), "gan_conv_1_d")
                .addLayer("gan_conv_2_d", new ConvolutionLayer.Builder(5, 5)
                        .stride(2, 2)
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        .nIn(64)
                        .nOut(128)
                        .build(), "gan_maxpool_1")
                .addLayer("gan_maxpool_2", new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(1, 1)
                        .build(), "gan_conv_2_d")
                .addLayer("gan_dense_d", new DenseLayer.Builder()
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        .nOut(1024)
                        .build(), "gan_maxpool_2")
                .addLayer("gan_output", new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .updater(new RmsProp(_frozen_learning_rate, 1e-8, 1e-8))
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .build(), "gan_dense_d")
                .setOutputs("gan_output")
                .build());
        gan.init();
        // Выводим сводное описание сети.
        System.out.println(gan.summary());

        return new ImmutableTriple<>(dis, gen, gan);
    }

    public DataSetIterator prepareData()
    {
        // Создаём итератор, читающий изображения из файлов и
        // преобразующий их в двумерные массивы.

        return new DataSetIterator()
        {
            public int epochCounter = 0;

            private int _currentBatch = 0;
            private final String trainingPath = _resPath + _dataSetName + "\\trainingData\\%d.jpg";
            private Integer[] _nums;

            @Override
            public DataSet next(int num)
            {
                return next();
            }

            @Override
            public int inputColumns()
            {
                return 0;
            }

            @Override
            public int totalOutcomes()
            {
                return 0;
            }

            @Override
            public boolean resetSupported()
            {
                return true;
            }

            @Override
            public boolean asyncSupported()
            {
                return false;
            }

            @Override
            public void reset()
            {
                // В конце эпохи случайным образом перемешиваем изображения.

                _currentBatch = 0;
                System.out.println(String.format("Эпоха %d завершена.", epochCounter));
                epochCounter++;
                Collections.shuffle(Arrays.asList(_nums));
            }

            @Override
            public int batch()
            {
                return _currentBatch;
            }

            @Override
            public void setPreProcessor(DataSetPreProcessor preProcessor)
            {

            }

            @Override
            public DataSetPreProcessor getPreProcessor()
            {
                return null;
            }

            @Override
            public List<String> getLabels()
            {
                return null;
            }

            @Override
            public boolean hasNext()
            {
                return true;
            }

            @Override
            public DataSet next()
            {
                // Считываем следующие _batchSize изображений,
                // преобразуем их в массив и возвращаем.

                try
                {
                    float[][][][] fData = new float[_batchSize][_channels][_width][_height];
                    for (int b = 0; b < _batchSize; b++, _currentBatch++)
                    {
                        if (_currentBatch >= _numTrainingSamples)
                        {
                            reset();
                        }
                        BufferedImage img = ImageIO.read(new File(String.format(trainingPath, _nums[_currentBatch])));
                        int[] data = img.getRGB(0, 0, _width, _height, null, 0, _width);
                        for (int i = 0; i < _width; i++)
                        {
                            for (int j = 0; j < _height; j++)
                            {
                                int pixel = data[i * _height + j];
                                if (_channels == 1)
                                {
                                    float value = (((pixel & 0x00ff0000) >> 16) +
                                            ((pixel & 0x0000ff00) >> 8) +
                                            (pixel & 0x000000ff)) / 3f;
                                    fData[b][0][i][j] = value / 255f;
                                }
                                else if (_channels == 3)
                                {
                                    fData[b][0][i][j] = ((pixel & 0x00ff0000) >> 16) / 255f;
                                    fData[b][1][i][j] = ((pixel & 0x0000ff00) >> 8) / 255f;
                                    fData[b][2][i][j] = (pixel & 0x000000ff) / 255f;
                                }
                                else
                                {
                                    fData[b][0][i][j] = ((pixel & 0xff000000) >> 24) / 255f;
                                    fData[b][1][i][j] = ((pixel & 0x00ff0000) >> 16) / 255f;
                                    fData[b][2][i][j] = ((pixel & 0x0000ff00) >> 8) / 255f;
                                    fData[b][3][i][j] = (pixel & 0x000000ff) / 255f;
                                }
                            }
                        }
                    }
                    INDArray result = Nd4j.create(fData);
                    return new DataSet(result, Nd4j.ones(1));
                }
                catch (Exception ex)
                {
                    return null;
                }
            }

            public DataSetIterator init()
            {
                _nums = new Integer[_numTrainingSamples];
                for (int i = 0; i < _numTrainingSamples; i++)
                {
                    _nums[i] = i;
                }
                Collections.shuffle(Arrays.asList(_nums));
                return this;
            }
        }.init();
    }

    public BufferedImage create()
    {
        // Генерируем изображение.

        INDArray out = _gen.output(Nd4j.rand(1, _genInputCount))[0];
        BufferedImage img = _prepareOutput(out.slice(0, 0), 1);
        return img;
    }

    public void create(String fileName) throws IOException
    {
        // Генерируем изображение и сохраняем в файл.

        BufferedImage img = create();
        File imgFile = new File(_resPath + _dataSetName + "\\" + fileName);
        imgFile.mkdirs();
        imgFile.createNewFile();
        imgFile.setWritable(true);
        ImageIO.write(img, "png", imgFile);
    }

    public void train(DataSetIterator iterTrain) throws IOException
    {
        // Обучаем модель.

        // Создаём вход генератора для тестов.
        INDArray testInput = Nd4j.rand(_numGenSamples * _numGenSamples, _genInputCount);

        // Проверяем начальную точность распознавания.
        ImmutableTriple<Float, Float, Float> startTestRes = _testModel();
        System.out.println(String.format("Средняя разница между оценкой реальных и сгенерированных данных: %f " +
                        "(средняя оценка реальных данных: %f; " +
                        "средняя оценка сгенерированных данных: %f).",
                startTestRes.getThird(), startTestRes.getFirst(), startTestRes .getSecond()));

        // Отключаем периодическую очистку памяти (для увеличения быстродействия).
        Nd4j.getMemoryManager().togglePeriodicGc(false);

        // Обучение до конца эпохи.
        for (int batch_counter = _trainingStartBatchNumber; batch_counter < _numIterations; batch_counter++)
        {
            INDArray soften_labels = Nd4j.ones(_batchSize, 1).mul(0.1);

            DataSet trDataSet = iterTrain.next();

            // Случайные числа в диапазоне [-1, 1) - вход генератора для обучения.
            INDArray randomInput = Nd4j.rand(_batchSize, _genInputCount).muli(2.0).subi(1.0);

            // Создаём для дискриминатора обучающие данные:
            // реальные изображения с меткой 0.9 и сгенерированные изображения с меткой 0.1.
            INDArray featuresArr = Nd4j.concat(0, trDataSet.getFeatures(), _gen.output(randomInput)[0]);
            INDArray labelsArr = Nd4j.concat(0, Nd4j.ones(_batchSize, 1).subi(soften_labels),
                    Nd4j.zeros(_batchSize, 1).addi(soften_labels));
            DataSet trainData = new DataSet(featuresArr, labelsArr);

            // Тренируем дискриминатор.
            _dis.fit(trainData);

            _updateGan();

            // Случайные числа в диапазоне [-1, 1) - вход генератора для обучения.
            randomInput = Nd4j.rand(_batchSize, _genInputCount).muli(2.0).subi(1.0);
            // Считая, что из случайного входа мы хотим получить реальные изображения...
            trainData = new DataSet(randomInput, Nd4j.ones(_batchSize, 1).subi(soften_labels));

            // ...обучаем генератор с прикрученной к нему необучаемой копией дискриминатора.
            _gan.fit(trainData);

            System.out.println(String.format("Батч №%d завершён.", batch_counter));

            // Сохраняем модель.
            if ((batch_counter % _saveEvery) == 0)
            {
                _updateGen();
                save(false);
            }

            // Генерируем и выводим изображения по входу для тестов.
            if ((batch_counter % _printEvery) == 0)
            {
                _updateGen();

                // Генерируем _numGenSamples * _numGenSamples изображений.
                INDArray out = _gen.output(testInput)[0];
                // Сохраняем их в файл в виде квадратной сетки.
                BufferedImage img = _prepareOutput(out, _numGenSamples);
                String fileName = String.format("generated\\batch%d.png", batch_counter);
                File imgFile = new File(_resPath + _dataSetName + "\\" + fileName);
                imgFile.mkdirs();
                imgFile.createNewFile();
                imgFile.setWritable(true);
                ImageIO.write(img, "png", imgFile);

                // Проверяем точность распознавания.
                ImmutableTriple<Float, Float, Float> testResults = _testModel();
                String strTestResult = String.format("Средняя разница между оценкой реальных и сгенерированных данных: %f " +
                                "(средняя оценка реальных данных: %f; " +
                                "средняя оценка сгенерированных данных: %f, батч: %d).",
                        testResults.getThird(), testResults.getFirst(), testResults.getSecond(), batch_counter);

                // Записываем точность в лог-файл.
                FileWriter writer = new FileWriter(_resPath + _dataSetName + "\\generated\\results.log", true);
                writer.write(strTestResult + "\n");
                writer.flush();
                writer.close();
                System.out.println(strTestResult);
            }
        }
        _updateGen();

        Nd4j.getMemoryManager().togglePeriodicGc(true);

        // Сохраняем итоги обучения.
        save(false);
    }

    private ImmutableTriple<Float, Float, Float> _testModel() throws IOException
    {
        // Вычисляем среднюю оценку (от 0 до 1) для сгенерированных
        // и реальных изображений, а также их разность.

        // Загружаем несколько реальных изображений.
        float[][][][] fData = new float[_numTestSamples][_channels][_width][_height];
        for (int b = 0; b < _numTestSamples; b++)
        {
            BufferedImage img = ImageIO.read(new File(String.format(_resPath + _dataSetName + "\\testData\\%d.jpg", b)));
            int[] data = img.getRGB(0, 0, _width, _height, null, 0, _width);
            for (int i = 0; i < _width; i++)
            {
                for (int j = 0; j < _height; j++)
                {
                    int pixel = data[i * _height + j];
                    if (_channels == 1)
                    {
                        float value = (((pixel & 0x00ff0000) >> 16) +
                                ((pixel & 0x0000ff00) >> 8) +
                                (pixel & 0x000000ff)) / 3f;
                        fData[b][0][i][j] = value / 255f;
                    }
                    else if (_channels == 3)
                    {
                        fData[b][0][i][j] = ((pixel & 0x00ff0000) >> 16) / 255f;
                        fData[b][1][i][j] = ((pixel & 0x0000ff00) >> 8) / 255f;
                        fData[b][2][i][j] = (pixel & 0x000000ff) / 255f;
                    }
                    else
                    {
                        fData[b][0][i][j] = ((pixel & 0xff000000) >> 24) / 255f;
                        fData[b][1][i][j] = ((pixel & 0x00ff0000) >> 16) / 255f;
                        fData[b][2][i][j] = ((pixel & 0x0000ff00) >> 8) / 255f;
                        fData[b][3][i][j] = (pixel & 0x000000ff) / 255f;
                    }
                }
            }
        }

        // Оцениваем реальные и сгенерированные изображения.
        float[] realResults = _dis.output(Nd4j.create(fData))[0].toFloatVector();
        float[] fakeResults = _gan.output(Nd4j.rand(_numTestSamples, _genInputCount))[0].toFloatVector();
        float realResult = 0;
        float fakeResult = 0;
        float result = 0;
        for (int i = 0; i < _numTestSamples; i++)
        {
            fakeResult += fakeResults[i];
            realResult += realResults[i];
            result += Math.abs(realResults[i] - fakeResults[i]);
        }
        return new ImmutableTriple<>(realResult / _numTestSamples, fakeResult / _numTestSamples, result / _numTestSamples);
    }

    private BufferedImage _prepareOutput(INDArray out, int sideCount)
    {
        // Преобразуем выход генератора в изображение.

        out = out.muli(0xff);
        int sideLength = sideCount * (_width + 1) - 1;
        BufferedImage img = new BufferedImage(sideLength, sideLength, BufferedImage.TYPE_INT_ARGB);

        // Записываем в файл каждое из sideCount * sideCount сгенерированных изображений.
        for (int d = 0; d < sideCount; d++)
        {
            for (int c = 0; c < sideCount; c++)
            {
                INDArray slice = out.slice(d * sideCount + c, 0);
                if (_channels == 1)
                {
                    // Для изображения в оттенках серого.

                    int[] intOut = slice.reshape(_flatSize).toIntVector();
                    int[] pixels = new int[_flatSize];
                    for (int j = 0; j < _flatSize; j++)
                    {
                        pixels[j] = (0xff << 24) | (intOut[j] << 16) | (intOut[j] << 8) | intOut[j];
                    }
                    img.setRGB(c * (_width + 1), d * (_height + 1), _width, _height, pixels, 0, _width);
                }
                else if (_channels == 3)
                {
                    // Для RGB-модели изображения.

                    int[] r = slice.slice(0, 0).reshape(_flatSize).toIntVector();
                    int[] g = slice.slice(1, 0).reshape(_flatSize).toIntVector();
                    int[] b = slice.slice(2, 0).reshape(_flatSize).toIntVector();
                    int[] pixels = new int[_flatSize];
                    for (int j = 0; j < _flatSize; j++)
                    {
                        pixels[j] = (0xff << 24) | (r[j] << 16) | (g[j] << 8) | b[j];
                    }
                    img.setRGB(c * (_width + 1), d * (_height + 1), _width, _height, pixels, 0, _width);
                }
                else
                {
                    // Для ARGB-модели изображения с прозрачностью.

                    int[] a = slice.slice(0, 0).reshape(_flatSize).toIntVector();
                    int[] r = slice.slice(1, 0).reshape(_flatSize).toIntVector();
                    int[] g = slice.slice(2, 0).reshape(_flatSize).toIntVector();
                    int[] b = slice.slice(3, 0).reshape(_flatSize).toIntVector();
                    int[] pixels = new int[_flatSize];
                    for (int j = 0; j < _flatSize; j++)
                    {
                        pixels[j] = (a[j] << 24) | (r[j] << 16) | (g[j] << 8) | b[j];
                    }
                    img.setRGB(c * (_width + 1), d * (_height + 1), _width, _height, pixels, 0, _width);
                }
            }
        }
        return img;
    }

    private void _updateGan()
    {
        // Копируем параметры частично обученного дискриминатора в тот,
        // который используется для обучения генератора.

        _gan.getLayer("gan_batch_norm_d").setParam("gamma", _dis.getLayer("dis_batch_norm").getParam("gamma"));
        _gan.getLayer("gan_batch_norm_d").setParam("beta",  _dis.getLayer("dis_batch_norm").getParam("beta"));
        _gan.getLayer("gan_batch_norm_d").setParam("mean",  _dis.getLayer("dis_batch_norm").getParam("mean"));
        _gan.getLayer("gan_batch_norm_d").setParam("var",   _dis.getLayer("dis_batch_norm").getParam("var"));

        _gan.getLayer("gan_conv_1_d").setParam("W",   _dis.getLayer("dis_conv_1").getParam("W"));
        _gan.getLayer("gan_conv_1_d").setParam("b",   _dis.getLayer("dis_conv_1").getParam("b"));

        _gan.getLayer("gan_conv_2_d").setParam("W",   _dis.getLayer("dis_conv_2").getParam("W"));
        _gan.getLayer("gan_conv_2_d").setParam("b",   _dis.getLayer("dis_conv_2").getParam("b"));

        _gan.getLayer("gan_dense_d").setParam("W",    _dis.getLayer("dis_dense").getParam("W"));
        _gan.getLayer("gan_dense_d").setParam("b",    _dis.getLayer("dis_dense").getParam("b"));

        _gan.getLayer("gan_output").setParam("W",   _dis.getLayer("dis_output").getParam("W"));
        _gan.getLayer("gan_output").setParam("b",   _dis.getLayer("dis_output").getParam("b"));
    }

    private void _updateGen()
    {
        // Копируем параметры частично обученного генератора в тот,
        // который в дальнейшем будет использоваться для генерации в обученной сети.

        _gen.getLayer("gen_batch_norm_1").setParam("gamma",   _gan.getLayer("gan_batch_norm_1").getParam("gamma"));
        _gen.getLayer("gen_batch_norm_1").setParam("beta",    _gan.getLayer("gan_batch_norm_1").getParam("beta"));
        _gen.getLayer("gen_batch_norm_1").setParam("mean",    _gan.getLayer("gan_batch_norm_1").getParam("mean"));
        _gen.getLayer("gen_batch_norm_1").setParam("var",     _gan.getLayer("gan_batch_norm_1").getParam("var"));

        _gen.getLayer("gen_dense_1").setParam("W", _gan.getLayer("gan_dense_1").getParam("W"));
        _gen.getLayer("gen_dense_1").setParam("b", _gan.getLayer("gan_dense_1").getParam("b"));

        _gen.getLayer("gen_dense_2").setParam("W", _gan.getLayer("gan_dense_2").getParam("W"));
        _gen.getLayer("gen_dense_2").setParam("b", _gan.getLayer("gan_dense_2").getParam("b"));

        _gen.getLayer("gen_batch_norm_2").setParam("gamma",   _gan.getLayer("gan_batch_norm_2").getParam("gamma"));
        _gen.getLayer("gen_batch_norm_2").setParam("beta",    _gan.getLayer("gan_batch_norm_2").getParam("beta"));
        _gen.getLayer("gen_batch_norm_2").setParam("mean",    _gan.getLayer("gan_batch_norm_2").getParam("mean"));
        _gen.getLayer("gen_batch_norm_2").setParam("var",     _gan.getLayer("gan_batch_norm_2").getParam("var"));

        _gen.getLayer("gen_conv_1").setParam("W",      _gan.getLayer("gan_conv_1").getParam("W"));
        _gen.getLayer("gen_conv_1").setParam("b",      _gan.getLayer("gan_conv_1").getParam("b"));

        _gen.getLayer("gen_conv_2").setParam("W",      _gan.getLayer("gan_conv_2").getParam("W"));
        _gen.getLayer("gen_conv_2").setParam("b",      _gan.getLayer("gan_conv_2").getParam("b"));
    }

    public void save(boolean onlyGenerator) throws IOException
    {
        // Сохраняем модель: только генератор (для использования обученной модели)
        // или все сети (для дальнейшего обучения).

        System.out.println("Saving models...");
        if (!onlyGenerator)
        {
            ModelSerializer.writeModel(_dis, new File(_resPath + _dataSetName + "\\" + _dataSetName + "_dis_model.zip") , true);
            ModelSerializer.writeModel(_gan, new File(_resPath + _dataSetName + "\\" + _dataSetName + "_gan_model.zip"), true);
        }
        ModelSerializer.writeModel(_gen, new File(_resPath + _dataSetName + "\\" + _dataSetName + "_gen_model.zip"), !onlyGenerator);
    }

    private GAN(String setName, boolean loadModels) throws IOException
    {
        // Создаём или загружаем модель для работы с заданным датасетом.

        _dataSetName = setName;
        System.out.println("Текущий датасет: " + setName);
        if (setName.equals("mnist"))
        {
            _width = 28;
            _height = 28;
            _channels = 1;
            _genInputCount = 6;
            _numTrainingSamples = 71850;
            _numTestSamples = 50;
            _batchSize = 150;
            _numGenSamples = 7;
        }
        else
        {
            throw new IllegalArgumentException("Неверное имя датасета!");
        }
        _flatSize = _width * _height;
        _numIterations = 10 * (_numTrainingSamples / _batchSize + 1);

        System.out.println("Current backend: " + Nd4j.getBackend());
        Nd4j.getMemoryManager().setAutoGcWindow(5000);

        if (!loadModels)
        {
            ImmutableTriple<ComputationGraph, ComputationGraph, ComputationGraph> networks = _createNetworks();
            _dis = networks.getFirst();
            _gen = networks.getSecond();
            _gan = networks.getThird();
        }
        else
        {
            _dis = ModelSerializer.restoreComputationGraph(new File(_resPath + _dataSetName + "\\" + _dataSetName + "_dis_model.zip"));
            _gen = ModelSerializer.restoreComputationGraph(new File(_resPath + _dataSetName + "\\" + _dataSetName + "_gen_model.zip"));
            _gan = ModelSerializer.restoreComputationGraph(new File(_resPath + _dataSetName + "\\" + _dataSetName + "_gan_model.zip"));
        }
    }
}