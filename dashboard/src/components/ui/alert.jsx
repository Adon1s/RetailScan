// dashboard/src/components/ui/alert.jsx
import React from 'react';

export function Alert({ children, className = '', ...props }) {
  return (
    <div className={`alert ${className}`} role="alert" {...props}>
      {children}
    </div>
  );
}

export function AlertDescription({ children, className = '', ...props }) {
  return (
    <div className={`alert-description ${className}`} {...props}>
      {children}
    </div>
  );
}