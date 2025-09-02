import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class QuizService {

  constructor() { }

  // Method to transform the Flask API response into a Quiz request body
  transformFlaskResponseToQuizRequest(flaskResponse: any): any {
    const quizRequest = {
      name: 'Osgood Connotation Theory Quiz',  // You can modify this as required
      summary: flaskResponse['6.pdf_query1'].query, // Assuming we are using the query as the summary
      questions: flaskResponse['6.pdf_query1']['correct answers'].map((correctAnswer: any) => {
        return {
          question: flaskResponse['6.pdf_query1'].query,
          options: flaskResponse['6.pdf_query1']['generated options'].map((option: any) => option.answer),
          answers: correctAnswer.answer
        };
      })
    };
    return quizRequest;
  }
}
