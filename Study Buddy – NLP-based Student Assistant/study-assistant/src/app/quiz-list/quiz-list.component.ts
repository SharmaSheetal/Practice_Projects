import { Component, OnInit } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-quiz-list',
  imports:[CommonModule, HttpClientModule],
  templateUrl: './quiz-list.component.html',
  styleUrls: ['./quiz-list.component.css']
})
export class QuizListComponent implements OnInit {

  quizzes: any[] = [];

  constructor(private http: HttpClient, private router: Router) {}

  ngOnInit() {
    this.loadQuizzes();
  }

  loadQuizzes() {
    this.http.get('http://localhost:8080/api/v1/quiz/quizzes').subscribe((data: any) => {
      this.quizzes = this.processQuizzes(data); 
      console.log(this.quizzes);
    });
  }

  startQuiz(quizId: string) {
    this.router.navigate(['/quiz', quizId]); // Navigate to quiz page with quizId
  }

  processQuizzes(quizzes: any[]) {
    const processedQuizzes: any[] = [];
    const quizNamesMap = new Map();  // Map to track the version number for each quiz name

    quizzes.forEach((quiz: any) => {
      const quizName = quiz.name;
      const version = quizNamesMap.has(quizName) ? quizNamesMap.get(quizName) + 1 : 1;
      quizNamesMap.set(quizName, version);
      quiz.version = version;  // Add version to the quiz data
      processedQuizzes.push(quiz);
    });

    return processedQuizzes;
  }
}
