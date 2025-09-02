import { Component } from '@angular/core';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { Question } from '../../models/question.model';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-flashcard',
  standalone: true,
  imports: [HttpClientModule, CommonModule],
  templateUrl: './flashcard.component.html',
  styleUrls: ['./flashcard.component.css'],
})
export class FlashcardComponent {
  questions: Question[] = [];
  currentIndex: number = 0;
  showAnswer: boolean = false;
  quizzes: any[] = [];
  selectedQuizId: string | null = null;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.loadQuizzes();
  }

  loadQuizzes() {
    this.http.get('http://localhost:8080/api/v1/quiz/quizzes').subscribe(
      (data: any) => {
        this.quizzes = this.processQuizzes(data);
      },
      (error) => {
        console.error('Error fetching quizzes:', error);
      }
    );
  }

  processQuizzes(quizzes: any[]) {
    const processedQuizzes: any[] = [];
    const quizNamesMap = new Map();

    quizzes.forEach((quiz: any) => {
      const quizName = quiz.name;
      const version = quizNamesMap.has(quizName)
        ? quizNamesMap.get(quizName) + 1
        : 1;
      quizNamesMap.set(quizName, version);
      quiz.version = version;
      processedQuizzes.push(quiz);
    });

    return processedQuizzes;
  }

  startFlashcards(quizId: string) {
    this.selectedQuizId = quizId;
    this.fetchQuestions(quizId);
  }

  fetchQuestions(quizId: string) {
    this.http
      .get(`http://localhost:8080/api/v1/quiz/${quizId}`)
      .subscribe((data: any) => {
        this.questions = data.questions || [];
        this.currentIndex = 0;
        this.showAnswer = false;
      });
  }

  toggleAnswer() {
    this.showAnswer = !this.showAnswer;
  }

  previousCard() {
    if (this.currentIndex > 0) {
      this.currentIndex--;
      this.showAnswer = false;
    }
  }

  nextCard() {
    if (this.currentIndex < this.questions.length - 1) {
      this.currentIndex++;
      this.showAnswer = false;
    }
  }

  goBackToQuizzes() {
    this.selectedQuizId = null;
    this.questions = [];
    this.currentIndex = 0;
    this.showAnswer = false;
  }
}
