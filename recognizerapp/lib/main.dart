import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:signature/signature.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Handwritten Character Recognizer',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(useMaterial3: true),
      home: const RecognitionScreen(),
    );
  }
}

class RecognitionScreen extends StatefulWidget {
  const RecognitionScreen({super.key});

  @override
  State<RecognitionScreen> createState() => _RecognitionScreenState();
}

class _RecognitionScreenState extends State<RecognitionScreen> {
  // 1 = digit,2 = alphabet
  int _value = 1;
  //for model loading
  Interpreter? _digitInterpreter;
  Interpreter? _alphaInterpreter;

  final TextEditingController _outputController = TextEditingController();

  final SignatureController _signatureController = SignatureController(
    penStrokeWidth: 10,
    penColor: Colors.black,
    exportBackgroundColor: Colors.white,
  );

  bool _isLoadingModel = false;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  @override
  void dispose() {
    _digitInterpreter?.close();
    _alphaInterpreter?.close();
    _signatureController.dispose();
    _outputController.dispose();
    super.dispose();
  }

  // model loading
  Future<void> loadModel() async {
    setState(() {
      _isLoadingModel = true;
    });

    try {
      if (_value == 1) {
        _digitInterpreter?.close();
        _digitInterpreter = await Interpreter.fromAsset(
          "assets/mnist_digit_model.tflite",
        );
        _outputController.text = "Digit model loaded";
      } else {
        _alphaInterpreter?.close();
        _alphaInterpreter = await Interpreter.fromAsset(
          "assets/emnist_alphabet_model.tflite",
        );
        _outputController.text = "Alphabet model loaded";
      }
    } catch (e) {
      _outputController.text = "Error loading model: $e";
    }

    setState(() {
      _isLoadingModel = false;
    });
  }

  //convert canvas drawing to png image in byte format
  Future<Uint8List?> _exportCanvasPngBytes() async {
    if (_signatureController.isEmpty) {
      _outputController.text = "Please draw something first";
      return null;
    }
    return await _signatureController.toPngBytes();
  }

  //preprocess for alphabet

  List<List<List<List<double>>>> _getCanvasInputTensorAlphabet(
    Uint8List pngBytes,
  ) {
    img.Image? decoded = img.decodeImage(pngBytes);
    if (decoded == null) throw Exception("Cannot decode image");

    // grayscale
    img.Image gray = img.grayscale(decoded);

    // bounding box and crop
    int minX = gray.width, minY = gray.height, maxX = 0, maxY = 0;
    bool foundInk = false;

    for (int y = 0; y < gray.height; y++) {
      for (int x = 0; x < gray.width; x++) {
        final p = gray.getPixel(x, y).r.toInt();

        if (p < 240) {
          foundInk = true;
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }

    img.Image cropped = gray;
    if (foundInk) {
      cropped = img.copyCrop(
        gray,
        x: minX,
        y: minY,
        width: (maxX - minX + 1),
        height: (maxY - minY + 1),
      );
    }

    // Add padding
    final pad = 20;
    img.Image padded = img.Image(
      width: cropped.width + pad * 2,
      height: cropped.height + pad * 2,
    );

    // white background
    img.fill(padded, color: img.ColorRgb8(255, 255, 255));
    img.compositeImage(padded, cropped, dstX: pad, dstY: pad);

    // Resize to 28x28
    img.Image resized = img.copyResize(
      padded,
      width: 28,
      height: 28,
      interpolation: img.Interpolation.linear,
    );

    // flip and rotate
    resized = img.copyRotate(resized, angle: 90);
    resized = img.flipHorizontal(resized);

    // normalize
    final input = List.generate(
      1,
      (_) => List.generate(
        28,
        (y) => List.generate(28, (x) {
          final px = resized.getPixel(x, y).r.toDouble() / 255.0;
          final inverted = 1.0 - px;
          return [inverted];
        }),
      ),
    );

    return input;
  }

  // preprocess for digits
  List<List<List<List<double>>>> _getCanvasInputTensorDigit(
    Uint8List pngBytes,
  ) {
    img.Image? decoded = img.decodeImage(pngBytes);
    if (decoded == null) throw Exception("Cannot decode canvas image");

    // grayscale
    img.Image gray = img.grayscale(decoded);

    // find bounding box
    int minX = gray.width, minY = gray.height, maxX = 0, maxY = 0;
    bool foundInk = false;

    for (int y = 0; y < gray.height; y++) {
      for (int x = 0; x < gray.width; x++) {
        int pixel = gray.getPixel(x, y).r.toInt();
        if (pixel < 240) {
          foundInk = true;
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }
    }

    img.Image cropped = gray;
    if (foundInk) {
      cropped = img.copyCrop(
        gray,
        x: minX,
        y: minY,
        width: (maxX - minX + 1),
        height: (maxY - minY + 1),
      );
    }

    // padding
    int pad = 8;
    img.Image padded = img.Image(
      width: cropped.width + pad * 2,
      height: cropped.height + pad * 2,
    );

    // fill background white
    img.fill(padded, color: img.ColorRgb8(255, 255, 255));

    img.compositeImage(padded, cropped, dstX: pad, dstY: pad);

    // resize to 28x28
    img.Image resized = img.copyResize(
      padded,
      width: 28,
      height: 28,
      interpolation: img.Interpolation.linear,
    );

    // 5) normalize + invert
    var input = List.generate(
      1,
      (_) => List.generate(
        28,
        (y) => List.generate(28, (x) {
          double pixel = resized.getPixel(x, y).r.toDouble() / 255.0;
          double inverted = 1.0 - pixel;
          return [inverted];
        }),
      ),
    );

    return input;
  }

  // digit recognition ()
  Future<void> recognizeDigit() async {
    if (_digitInterpreter == null) {
      _outputController.text = "Digit model not loaded";
      return;
    }

    final pngBytes = await _exportCanvasPngBytes();
    if (pngBytes == null) return;

    try {
      final input = _getCanvasInputTensorDigit(pngBytes);

      // output shape [1,10]
      final output = List.generate(1, (_) => List.filled(10, 0.0));

      _digitInterpreter!.run(input, output);

      double bestScore = output[0][0];
      int bestIndex = 0;
      for (int i = 1; i < 10; i++) {
        if (output[0][i] > bestScore) {
          bestScore = output[0][i];
          bestIndex = i;
        }
      }

      setState(() {
        _outputController.text =
            "Digit: $bestIndex  (confidence: ${(bestScore * 100).toStringAsFixed(2)}%)";
      });
    } catch (e) {
      _outputController.text = "Digit recognize error: $e";
    }
  }

  // alphabet recognition ()
  Future<void> recognizeAlphabet() async {
    if (_alphaInterpreter == null) {
      _outputController.text = "Alphabet model not loaded";
      return;
    }

    final pngBytes = await _exportCanvasPngBytes();
    if (pngBytes == null) return;

    try {
      final input = _getCanvasInputTensorAlphabet(pngBytes);

      // for shape [1,26]
      final output = List.generate(1, (_) => List.filled(26, 0.0));

      _alphaInterpreter!.run(input, output);

      double bestScore = output[0][0];
      int bestIndex = 0;
      for (int i = 1; i < 26; i++) {
        if (output[0][i] > bestScore) {
          bestScore = output[0][i];
          bestIndex = i;
        }
      }

      String predictedLetter = String.fromCharCode(65 + bestIndex);

      setState(() {
        _outputController.text =
            "Alphabet: $predictedLetter  (confidence: ${(bestScore * 100).toStringAsFixed(2)}%)";
      });
    } catch (e) {
      _outputController.text = "Alphabet recognize error: $e";
    }
  }

  Future<void> recognize() async {
    if (_value == 1) {
      await recognizeDigit();
    } else {
      await recognizeAlphabet();
    }
  }

  // User interface
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Handwritten Character Recognizer"),
        backgroundColor: Colors.blue,
        foregroundColor: Colors.white,
      ),
      body: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          children: [
            Container(
              width: MediaQuery.of(context).size.width * 0.9,
              height: MediaQuery.of(context).size.height * 0.45,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: Colors.grey.shade400),
              ),
              child: Signature(
                controller: _signatureController,
                backgroundColor: Colors.white,
              ),
            ),

            const SizedBox(height: 12),

            Align(
              alignment: Alignment.centerRight,
              child: ElevatedButton.icon(
                onPressed: () {
                  _signatureController.clear();
                  setState(() {
                    _outputController.text = "";
                  });
                },
                icon: const Icon(Icons.clear),
                label: const Text("Clear"),
              ),
            ),

            const SizedBox(height: 10),

            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Radio(
                  value: 1,
                  groupValue: _value,
                  activeColor: Colors.blue,
                  onChanged: (val) async {
                    setState(() {
                      _value = val!;
                    });
                    await loadModel();
                  },
                ),
                const Text("Digit"),
                const SizedBox(width: 20),
                Radio(
                  value: 2,
                  groupValue: _value,
                  activeColor: Colors.blue,
                  onChanged: (val) async {
                    setState(() {
                      _value = val!;
                    });
                    await loadModel();
                  },
                ),
                const Text("Alphabet"),
              ],
            ),

            const SizedBox(height: 10),

            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: _isLoadingModel ? null : recognize,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  backgroundColor: Colors.blue,
                  foregroundColor: Colors.white,
                ),
                child: Text(_isLoadingModel ? "Loading model..." : "Recognize"),
              ),
            ),

            const SizedBox(height: 12),

            TextField(
              controller: _outputController,
              readOnly: true,
              decoration: InputDecoration(
                hintText: 'Output',
                filled: true,
                fillColor: Colors.grey.shade200,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide.none,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
