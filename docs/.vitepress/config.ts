import { defineConfig } from 'vitepress';
import { progressPlugin } from './markdown/progress-plugin';

export default defineConfig({
  title: 'Knowledge Hub: Developer Kit',
  description: 'A comprehensive engineering book (Foundations to Advanced)',
  lastUpdated: true,
  markdown: {
    config(md) {
      md.use(progressPlugin);
    }
  },
  themeConfig: {
    search: {
      provider: 'local'
    },
    outline: {
      level: [2, 3],
      label: 'On this page'
    },
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Study Plan', link: '/start-here/study-plan' },
      { text: 'Review', link: '/start-here/review' },
      { text: 'My Progress', link: '/start-here/progress' },
      { text: 'Quick Revision', link: '/start-here/quick-revision' },
      {
        text: 'Topics',
        items: [
          { text: 'JavaScript & TypeScript', link: '/javascript/01-core-language' },
          { text: 'React', link: '/react/00-preface' },
          { text: 'Node & Express', link: '/node-express/00-node-express-preface' },
          { text: 'System Design', link: '/system-design/00-interview-framework' },
          { text: 'DSA', link: '/dsa/01-patterns' },
          { text: 'Databases & SQL', link: '/databases/01-sql-fundamentals' },
          { text: 'Web Fundamentals', link: '/web-fundamentals/01-http-networking' },
          { text: 'Behavioral & Career', link: '/behavioral/01-story-bank' },
          { text: 'Core Concepts', link: '/core-concepts/01-scalability' },
          { text: 'AWS AI Practitioner', link: '/aws-ai-practitioner/' }
        ]
      }
    ],
    sidebar: [
      {
        text: 'Start Here',
        collapsed: false,
        items: [
          { text: '8-Week Study Plan', link: '/start-here/study-plan' },
          { text: 'Flashcard Review', link: '/start-here/review' },
          { text: 'My Progress', link: '/start-here/progress' },
          { text: 'Quick Revision Hub', link: '/start-here/quick-revision' },
          { text: 'Revision Tracker', link: '/start-here/revision-tracker' }
        ]
      },
      {
        text: 'JavaScript & TypeScript',
        collapsed: true,
        items: [
          { text: 'Core Language', link: '/javascript/01-core-language' },
          { text: 'Async & Event Loop', link: '/javascript/02-async-event-loop' },
          { text: 'Implement It Yourself', link: '/javascript/03-implement-yourself' },
          { text: 'TypeScript Essentials', link: '/javascript/04-typescript' },
          { text: 'Flashcards', link: '/javascript/flashcards' }
        ]
      },
      {
        text: 'React',
        collapsed: true,
        items: [
          { text: 'Preface', link: '/react/00-preface' },
          { text: 'Core Foundations', link: '/react/01-core-react-foundations' },
          {
            text: 'Rendering & Concurrency',
            link: '/react/02-rendering-concurrency'
          },
          { text: 'Hooks Fundamentals', link: '/react/03-hooks-fundamentals' },
          { text: 'Hooks Advanced', link: '/react/04-hooks-advanced' },
          {
            text: 'State Management Strategies',
            link: '/react/05-state-management-strategies'
          },
          {
            text: 'Redux Toolkit & Global State',
            link: '/react/06-redux-toolkit-global-state'
          },
          {
            text: 'Server State & TanStack Query',
            link: '/react/07-server-state-tanstack-query'
          },
          {
            text: 'Caching & Offline',
            link: '/react/08-caching-optimistic-offline'
          },
          {
            text: 'Performance Engineering',
            link: '/react/09-performance-engineering'
          },
          {
            text: 'Component Design Patterns',
            link: '/react/10-component-design-patterns'
          },
          {
            text: 'Accessibility & UX Quality',
            link: '/react/11-accessibility-ux-quality'
          },
          {
            text: 'Testing & Reliability',
            link: '/react/12-testing-reliability'
          },
          {
            text: 'Security & Robustness',
            link: '/react/13-security-robustness'
          },
          {
            text: 'Advanced Topics',
            link: '/react/14-advanced-topics-ssr-suspense-ts'
          },
          {
            text: 'Anti-Patterns & Refactoring',
            link: '/react/15-anti-patterns-refactoring'
          },
          {
            text: 'Night-Before Sheet & Flashcards',
            link: '/react/16-interview-cheatsheet'
          },
          { text: 'Recipes & Utilities', link: '/react/17-recipes-utilities' },
          {
            text: 'Glossary & Further Reading',
            link: '/react/18-glossary-further-reading'
          }
        ]
      },
      {
        text: 'React Q&A Deep Dives',
        collapsed: true,
        items: [
          { text: 'Design Patterns', link: '/qna/DESIGN_PATTERNS' },
          { text: 'Hooks Deep Dive', link: '/qna/Hooks_Deep_Dive' },
          { text: 'React 19', link: '/qna/REACT_19' },
          { text: 'React Mastery Guide', link: '/qna/REACT_MASTERY_GUIDE' },
          { text: 'Redux Toolkit', link: '/qna/REDUX_TOOLKIT' },
          { text: 'State Management', link: '/qna/STATE_MANAGEMENT' },
          { text: 'TanStack Query', link: '/qna/TANSTACK_QUERY' }
        ]
      },
      {
        text: 'Node & Express',
        collapsed: true,
        items: [
          { text: 'Preface', link: '/node-express/00-node-express-preface' },
          {
            text: 'Runtime Foundations',
            link: '/node-express/01-runtime-foundations'
          },
          {
            text: 'Event Loop Concurrency',
            link: '/node-express/02-event-loop-concurrency'
          },
          {
            text: 'Modules Packaging',
            link: '/node-express/03-modules-packaging'
          },
          { text: 'Async Patterns', link: '/node-express/04-async-patterns' },
          {
            text: 'Express Routing Middleware',
            link: '/node-express/05-express-routing-middleware'
          },
          {
            text: 'Security Hardening',
            link: '/node-express/06-security-hardening'
          },
          {
            text: 'Performance Profiling',
            link: '/node-express/07-performance-profiling'
          },
          {
            text: 'Testing Observability',
            link: '/node-express/08-testing-observability'
          },
          {
            text: 'Architecture Patterns',
            link: '/node-express/09-architecture-patterns'
          },
          {
            text: 'Deployment Scaling',
            link: '/node-express/10-deployment-scaling'
          },
          { text: 'Advanced Topics', link: '/node-express/11-advanced-topics' },
          {
            text: 'Night-Before Sheet & Flashcards',
            link: '/node-express/12-node-express-cheatsheet'
          },
          { text: 'Redis Caching', link: '/node-express/13-redis-caching' },
          {
            text: 'Express Error Handling',
            link: '/node-express/14-express-error-handling'
          },
          {
            text: 'Express Middleware Flow',
            link: '/node-express/15-express-middleware-flow'
          }
        ]
      },
      {
        text: 'System Design',
        collapsed: true,
        items: [
          { text: 'Interview Framework', link: '/system-design/00-interview-framework' },
          { text: 'Data Storage & Consistency', link: '/system-design/01-data-storage' },
          { text: 'Caching & CDN', link: '/system-design/02-caching-cdn' },
          { text: 'Messaging & Async', link: '/system-design/03-messaging-async' },
          {
            text: 'API Design & Rate Limiting',
            link: '/system-design/04-api-design-rate-limiting'
          },
          { text: 'Auth & Security', link: '/system-design/05-auth-security' },
          { text: 'Case Studies', link: '/system-design/06-case-studies' },
          {
            text: 'Frontend System Design',
            link: '/system-design/07-frontend-system-design'
          },
          { text: 'Flashcards', link: '/system-design/flashcards' }
        ]
      },
      {
        text: 'Core Concepts',
        collapsed: true,
        items: [
          { text: 'Scalability', link: '/core-concepts/01-scalability' },
          {
            text: 'High Availability & Fault Tolerance',
            link: '/core-concepts/02-high-availability-fault-tolerance'
          },
          {
            text: 'Microservices vs Monolith',
            link: '/core-concepts/03-microservices-vs-monolith'
          }
        ]
      },
      {
        text: 'DSA',
        collapsed: true,
        items: [
          { text: 'Patterns & Templates', link: '/dsa/01-patterns' },
          { text: 'JS Toolkit', link: '/dsa/02-js-toolkit' },
          { text: 'Problem Tracker', link: '/dsa/03-problem-tracker' },
          { text: 'Pattern Flashcards', link: '/dsa/flashcards' }
        ]
      },
      {
        text: 'Databases & SQL',
        collapsed: true,
        items: [
          { text: 'SQL & DB Fundamentals', link: '/databases/01-sql-fundamentals' },
          { text: 'SQL Exercises', link: '/databases/02-sql-exercises' },
          { text: 'Flashcards', link: '/databases/flashcards' }
        ]
      },
      {
        text: 'Web Fundamentals',
        collapsed: true,
        items: [
          { text: 'HTTP & Networking', link: '/web-fundamentals/01-http-networking' },
          {
            text: 'Browser, Performance & Security',
            link: '/web-fundamentals/02-browser-performance'
          },
          { text: 'Flashcards', link: '/web-fundamentals/flashcards' }
        ]
      },
      {
        text: 'Behavioral & Career',
        collapsed: true,
        items: [
          { text: 'STAR Story Bank', link: '/behavioral/01-story-bank' },
          {
            text: 'Intro & Project Deep Dive',
            link: '/behavioral/02-intro-and-project-deep-dive'
          },
          {
            text: 'Senior Signals & Offers',
            link: '/behavioral/03-senior-signals-and-offers'
          },
          { text: 'Job Search & Trackers', link: '/behavioral/04-job-search-tracker' }
        ]
      },
      {
        text: 'AWS AI Practitioner',
        collapsed: true,
        items: [
          { text: 'Study Guide', link: '/aws-ai-practitioner/' },
          { text: 'Overview', link: '/aws-ai-practitioner/overview' },
          {
            text: 'Overview - MCQ Practice',
            link: '/aws-ai-practitioner/overview-mcq'
          },
          {
            text: 'Domain 1: Fundamentals of AI and ML',
            link: '/aws-ai-practitioner/domain-1-fundamentals'
          },
          {
            text: 'Domain 2: Fundamentals of Generative AI',
            link: '/aws-ai-practitioner/domain-2-generative-ai'
          },
          {
            text: 'Domain 3: Applications of Foundation Models',
            link: '/aws-ai-practitioner/domain-3-applications'
          },
          {
            text: 'Domain 4: Guidelines for Responsible AI',
            link: '/aws-ai-practitioner/domain-4-responsible-ai'
          },
          {
            text: 'Domain 5: Security, Compliance and Governance',
            link: '/aws-ai-practitioner/domain-5-security-compliance'
          },
          {
            text: 'AWS AI/ML Services',
            link: '/aws-ai-practitioner/aws-services'
          }
        ]
      }
    ],
    socialLinks: [
      {
        icon: 'github',
        link: 'https://github.com/PsychoTakiya/developer-kit'
      }
    ]
  }
});
