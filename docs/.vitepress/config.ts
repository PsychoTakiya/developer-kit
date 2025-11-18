import { defineConfig } from 'vitepress';

export default defineConfig({
  title: 'Knowledge Hub: Developer Kit',
  description: 'A comprehensive engineering book (Foundations to Advanced)',
  lastUpdated: true,
  themeConfig: {
    outline: {
      level: [2, 3],
      label: 'On this page'
    },
    nav: [
      { text: 'Guide', link: '/' },
      { text: 'Glossary', link: '/react/18-glossary-further-reading' }
    ],
    sidebar: [
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
      { text: 'Testing & Reliability', link: '/react/12-testing-reliability' },
      { text: 'Security & Robustness', link: '/react/13-security-robustness' },
      {
        text: 'Advanced Topics',
        link: '/react/14-advanced-topics-ssr-suspense-ts'
      },
      {
        text: 'Anti-Patterns & Refactoring',
        link: '/react/15-anti-patterns-refactoring'
      },
      {
        text: 'QNA - React Deep Dives',
        items: [
          {
            text: 'Design Patterns',
            link: '/qna/DESIGN_PATTERNS'
          },
          {
            text: 'Hooks Deep Dive',
            link: '/qna/Hooks_Deep_Dive'
          },
          {
            text: 'React 19',
            link: '/qna/REACT_19'
          },
          {
            text: 'React Mastery Guide',
            link: '/qna/REACT_MASTERY_GUIDE'
          },
          {
            text: 'Redux Toolkit',
            link: '/qna/REDUX_TOOLKIT'
          },
          {
            text: 'State Management',
            link: '/qna/STATE_MANAGEMENT'
          },
          {
            text: 'TanStack Query',
            link: '/qna/TANSTACK_QUERY'
          }
        ]
      },
      {
        text: 'Core Concepts',
        items: [
          {
            text: 'Scalability',
            link: '/core-concepts/01-scalability'
          },
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
        text: 'Node & Express',
        items: [
          {
            text: 'Node & Express Preface',
            link: '/node-express/00-node-express-preface'
          },
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
          {
            text: 'Async Patterns',
            link: '/node-express/04-async-patterns'
          },
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
          {
            text: 'Advanced Topics',
            link: '/node-express/11-advanced-topics'
          },
          {
            text: 'Node Express Cheatsheet',
            link: '/node-express/12-node-express-cheatsheet'
          },
          {
            text: 'Redis Caching',
            link: '/node-express/13-redis-caching'
          },
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
        text: 'AWS AI Practitioner',
        items: [
          {
            text: 'Overview',
            link: '/aws-ai-practitioner/overview'
          },
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
      },
      { text: 'Interview Cheat Sheet', link: '/react/16-interview-cheatsheet' },
      { text: 'Recipes & Utilities', link: '/react/17-recipes-utilities' },
      {
        text: 'Glossary & Further Reading',
        link: '/react/18-glossary-further-reading'
      }
    ],
    socialLinks: [
      {
        icon: 'github',
        link: 'https://github.com/your-org/react-developer-kit'
      }
    ]
  }
});
