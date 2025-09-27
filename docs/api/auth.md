# Authentication API

Authentication endpoints for user login, logout, and token management.

## POST /auth/login

Authenticate user and receive access tokens.

### Request

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

### Response

```json
{
  "status": "success",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "user_id": "admin_001",
    "username": "admin",
    "roles": ["admin", "user"],
    "expires_in": 3600
  },
  "message": "Login successful"
}
```

### Error Responses

```json
{
  "status": "error",
  "error": "Invalid credentials",
  "code": "AUTH_INVALID"
}
```

## POST /auth/logout

Logout user and invalidate tokens.

### Request

```http
POST /api/v1/auth/logout
Authorization: Bearer <access_token>
```

### Response

```json
{
  "status": "success",
  "message": "Logout successful"
}
```

## POST /auth/refresh

Refresh access token using refresh token.

### Request

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### Response

```json
{
  "status": "success",
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "expires_in": 3600
  },
  "message": "Token refreshed successfully"
}
```

## GET /auth/me

Get current user information.

### Request

```http
GET /api/v1/auth/me
Authorization: Bearer <access_token>
```

### Response

```json
{
  "status": "success",
  "data": {
    "user_id": "admin_001",
    "username": "admin",
    "email": "admin@example.com",
    "roles": ["admin", "user"],
    "permissions": [
      "scan_faces",
      "manage_users",
      "view_analytics"
    ],
    "created_at": "2023-01-01T00:00:00Z",
    "last_login": "2023-12-01T10:30:00Z"
  }
}
```

## POST /auth/change-password

Change user password.

### Request

```http
POST /api/v1/auth/change-password
Authorization: Bearer <access_token>
Content-Type: application/json

{
  "current_password": "oldpassword",
  "new_password": "newpassword"
}
```

### Response

```json
{
  "status": "success",
  "message": "Password changed successfully"
}
```

## POST /auth/reset-password

Request password reset (sends email).

### Request

```http
POST /api/v1/auth/reset-password
Content-Type: application/json

{
  "email": "user@example.com"
}
```

### Response

```json
{
  "status": "success",
  "message": "Password reset email sent"
}
```

## POST /auth/verify-token

Verify if token is valid.

### Request

```http
POST /api/v1/auth/verify-token
Content-Type: application/json

{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### Response

```json
{
  "status": "success",
  "data": {
    "valid": true,
    "user_id": "admin_001",
    "expires_at": "2023-12-01T11:30:00Z"
  }
}
```

## Authentication Flow

1. **Login**: Send username/password to `/auth/login`
2. **Store Tokens**: Save access_token and refresh_token
3. **API Calls**: Include access_token in Authorization header
4. **Token Refresh**: Use refresh_token to get new access_token when expired
5. **Logout**: Call `/auth/logout` to invalidate tokens

## Token Structure

### Access Token Payload
```json
{
  "user_id": "admin_001",
  "roles": ["admin", "user"],
  "type": "access",
  "exp": 1701426600,
  "iat": 1701423000
}
```

### Refresh Token Payload
```json
{
  "user_id": "admin_001",
  "type": "refresh",
  "exp": 1702031400,
  "iat": 1701423000
}
```

## Security Notes

- Access tokens expire in 1 hour
- Refresh tokens expire in 7 days
- Tokens are signed with HMAC-SHA256
- Passwords are hashed with SHA-256 + salt
- Rate limiting applies to login attempts
- Failed login attempts are logged for security monitoring
