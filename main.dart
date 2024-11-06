import 'package:http/http.dart' as http;
import 'dart:convert';

Future<void> getRecommendations(List<double> userFeatures) async {
  // Replace 'http://127.0.0.1:8000' with your Flask server URL if needed
  final url = Uri.parse('http://127.0.0.1:8000/get-recommendations/');
  
  // Prepare the request body
  final body = jsonEncode({"features": userFeatures});

  // Send POST request
  final response = await http.post(
    url,
    headers: {"Content-Type": "application/json"},
    body: body,
  );

  if (response.statusCode == 200) {
    // Parse the JSON response
    final jsonData = jsonDecode(response.body);
    print(jsonData);
  } else {
    print('Error: ${response.reasonPhrase}');
  }
}
