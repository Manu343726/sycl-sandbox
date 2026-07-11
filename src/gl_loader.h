#pragma once
// Minimal OpenGL 3.3 core function declarations.
//
// The system <GL/gl.h> only declares OpenGL 1.x functions. All OpenGL 3.x+
// core functions are exported by libGL.so but NOT declared in the headers.
// This header provides the missing extern "C" declarations so we can link
// against them directly.
//
// Include this instead of <GL/gl.h> / <GLFW/glfw3.h> — it provides both.

#include <GLFW/glfw3.h>
#include <cstddef>  // ptrdiff_t (used in place of GLsizeiptr)

extern "C" {

// ── Shader objects (OpenGL 2.0+ / core) ────────────────────────────────
extern GLuint            glCreateShader(GLenum type);
extern void              glShaderSource(GLuint shader, GLsizei count, const GLchar **string, const GLint *length);
extern void              glCompileShader(GLuint shader);
extern void              glGetShaderiv(GLuint shader, GLenum pname, GLint *params);
extern void              glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
extern GLuint            glCreateProgram(void);
extern void              glAttachShader(GLuint program, GLuint shader);
extern void              glLinkProgram(GLuint program);
extern void              glGetProgramiv(GLuint program, GLenum pname, GLint *params);
extern void              glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
extern void              glDeleteShader(GLuint shader);
extern void              glDeleteProgram(GLuint program);
extern void              glUseProgram(GLuint program);
extern GLint             glGetUniformLocation(GLuint program, const GLchar *name);
extern void              glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value);
extern void              glUniform3f(GLint location, GLfloat v0, GLfloat v1, GLfloat v2);
extern void              glUniform1i(GLint location, GLint v0);

// ── Vertex Array Objects (OpenGL 3.0+ / core) ──────────────────────────
extern void              glGenVertexArrays(GLsizei n, GLuint *arrays);
extern void              glBindVertexArray(GLuint array);
extern void              glDeleteVertexArrays(GLsizei n, const GLuint *arrays);

// ── Buffer objects (OpenGL 1.5+ / core) ────────────────────────────────
extern void              glGenBuffers(GLsizei n, GLuint *buffers);
extern void              glBindBuffer(GLenum target, GLuint buffer);
extern void              glBufferData(GLenum target, ptrdiff_t size, const void *data, GLenum usage);
extern void              glDeleteBuffers(GLsizei n, const GLuint *buffers);

// ── Vertex attributes (OpenGL 2.0+ / core) ─────────────────────────────
extern void              glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void *pointer);
extern void              glEnableVertexAttribArray(GLuint index);
extern void              glDisableVertexAttribArray(GLuint index);

// ── Drawing commands ───────────────────────────────────────────────────
extern void              glDrawArrays(GLenum mode, GLint first, GLsizei count);
extern void              glDrawElements(GLenum mode, GLsizei count, GLenum type, const void *indices);

// ── Framebuffer Objects (OpenGL 3.0+ / core) ───────────────────────────
extern void              glGenFramebuffers(GLsizei n, GLuint *framebuffers);
extern void              glBindFramebuffer(GLenum target, GLuint framebuffer);
extern void              glDeleteFramebuffers(GLsizei n, const GLuint *framebuffers);
extern GLenum            glCheckFramebufferStatus(GLenum target);
extern void              glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget, GLuint texture, GLint level);

// ── Renderbuffer Objects ───────────────────────────────────────────────
extern void              glGenRenderbuffers(GLsizei n, GLuint *renderbuffers);
extern void              glBindRenderbuffer(GLenum target, GLuint renderbuffer);
extern void              glRenderbufferStorage(GLenum target, GLenum internalformat, GLsizei width, GLsizei height);
extern void              glFramebufferRenderbuffer(GLenum target, GLenum attachment, GLenum renderbuffertarget, GLuint renderbuffer);
extern void              glDeleteRenderbuffers(GLsizei n, const GLuint *renderbuffers);

// ── GL state ───────────────────────────────────────────────────────────
extern void              glBlendFuncSeparate(GLenum sfactorRGB, GLenum dfactorRGB, GLenum sfactorAlpha, GLenum dfactorAlpha);
extern void              glLineWidth(GLfloat width);
extern void              glPointSize(GLfloat size);

} // extern "C"
