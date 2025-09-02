import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { SidebarNavigationComponent } from './sidebar-navigation/sidebar-navigation.component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, SidebarNavigationComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {
  title = 'study-assistant';
}
