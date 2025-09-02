import { RouterModule, Routes } from '@angular/router';
import { QuizComponent } from './quiz/quiz.component';
import { FlashcardComponent } from './flashcard/flashcard.component';
import { QuizListComponent } from './quiz-list/quiz-list.component';

import { LandingPageComponent } from './landing-page/landing-page.component';
import { SummaryPageComponent } from './summary-page/summary-page.component';
import { NgModule } from '@angular/core';
import { QuizResultsComponent } from './quiz-result/quiz-result.component';

export const routes: Routes = [
    { path: 'quiz/:quizId', component: QuizComponent },
    {path: 'flashcard', component: FlashcardComponent },
      { path: 'landing-page', component: LandingPageComponent },
    { path: 'summary-page', component: SummaryPageComponent },
    {path: 'quizzes', component:QuizListComponent},
    {path: 'results', component:QuizResultsComponent}
];
@NgModule({
    imports: [RouterModule.forRoot(routes)],
    exports: [RouterModule]
  })
  export class AppRoutingModule { }
